"""Strict loader: preserve episode boundaries and never guess missing grasp modes."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scripts.learning.policy_contract import encode_observation, normalize_action, MODES


@dataclass
class Episode:
    path: str
    mode: str
    observations: np.ndarray
    actions: np.ndarray


def load_episode(directory):
    directory = Path(directory)
    context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("success") is not True:
        raise ValueError("BC requires a successful episode")
    mode = context.get("grasp_mode")
    if mode not in MODES:
        raise ValueError("context.json must identify grasp_mode")
    if context.get("place_xy") != "measured TCP immediately before opening":
        raise ValueError("Unsupported place convention; expected measured world TCP XY")
    expected_pick = ("measured TCP at manual close" if mode == "physical"
                     else "measured cloth corner at grasp")
    if context.get("pick_xy") != expected_pick:
        raise ValueError("Unsupported pick convention")
    records = [json.loads(line) for line in (directory / "high_level.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    observations, actions = [], []
    with np.load(directory / "low_level.npz", allow_pickle=False) as low:
        tcp = low["tcp_w"]
        if tcp.ndim != 2 or tcp.shape[1] != 3 or not np.isfinite(tcp).all():
            raise ValueError("Invalid low-level TCP positions")
        for record in records:
            start, end = record["low_level_start"], record["low_level_end"]
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(tcp):
                raise ValueError("Invalid low-level segment")
            # Virtual legacy pick denotes a cloth corner. The first recorded TCP
            # after attachment supplies the actual measured gripper pick location.
            pick = tcp[start, :2] if mode == "virtual" else record["action"]["pick"]
            action = normalize_action([*pick, *record["action"]["place"]])
            observations.append(encode_observation(record["observation"]["corners"], context["pairs"], mode))
            actions.append(action)
    if not observations:
        raise ValueError("Episode has no pick/place transitions")
    return Episode(str(directory.resolve()), mode, np.stack(observations), np.stack(actions))


def load_episodes(roots):
    paths = sorted({p.parent.resolve() for root in roots for p in Path(root).rglob("high_level.jsonl")})
    episodes, rejected = [], []
    for path in paths:
        try:
            episodes.append(load_episode(path))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    return episodes, rejected


def split_episodes(episodes, validation_fraction=.2, seed=0):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(episodes) < 2:
        raise ValueError("Need at least two complete episodes for train/validation splitting")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    # Stratify by mode where possible, without splitting any single episode.
    for mode in MODES:
        group = [episode for episode in episodes if episode.mode == mode]
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * validation_fraction))) if len(group) > 1 else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    if not validation:
        raise ValueError("Need at least two episodes of one grasp mode for validation")
    return train, validation
