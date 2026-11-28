"""Load successful teleop demonstrations for separate high-level and low-level BC.

High-level: 30-D cloth observation -> normalized pick XY (2-D).
Low-level: 28-D previous state -> next absolute root-frame IK action (8-D).
Low-level actions are RAW here: train_bc.py must normalize them with saved bounds.

A recorded row is written AFTER its action was executed. Consequently the low-level
supervised sample uses observation at row t-1 and command from row t, never the
post-action observation at row t as the input for command t.

Correction pick/place records are accepted as additional high-level samples. Their
low-level pair index is read from the recorded pair_index rather than the record order.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scripts.learning.policy_contract import (
    MODES, encode_observation, encode_low_observation, normalize_action,
)

CARRY_PHASE = "carry"
RELEASE_PHASES = {"opening", "physical_release"}


@dataclass
class Episode:
    path: str
    mode: str
    observations: np.ndarray  # Legacy name: high-level (num_picks, 30)
    actions: np.ndarray       # Legacy name: high-level (num_picks, 2)
    low_observations: np.ndarray  # (num_low_steps, 28)
    low_actions: np.ndarray       # (num_low_steps, 8) unnormalized IK targets

    @property
    def high_observations(self):
        return self.observations

    @property
    def high_actions(self):
        return self.actions


def _read_json(folder, filename):
    return json.loads((folder / filename).read_text(encoding="utf-8"))


def _pairs(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("context.json must contain exactly two directed goal pairs")
    pairs = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(index) is not int for index in pair)):
            raise ValueError("Invalid goal pair indices")
        pairs.append(tuple(pair))
    if sorted(index for pair in pairs for index in pair) != [0, 1, 2, 3]:
        raise ValueError("Goal pairs must use each corner exactly once")
    return pairs


def _check_log(log):
    required = ("time_s", "phase", "action", "corners_w", "tcp_w", "grasping")
    missing = [field for field in required if field not in log]
    if missing:
        raise ValueError(f"low_level.npz missing: {missing}")
    time, phase, action, corners, tcp, grasping = (log[field] for field in required)
    n = len(time)
    expected = ((phase, (n,)), (action, (n, 8)), (corners, (n, 4, 3)),
                (tcp, (n, 3)), (grasping, (n,)))
    if any(value.shape != shape for value, shape in expected):
        raise ValueError("Invalid low_level.npz array shapes")
    if not all(np.isfinite(value).all() for value in (time, action, corners, tcp)):
        raise ValueError("Non-finite teleop data")
    if n > 1 and not np.all(np.diff(time) > 0):
        raise ValueError("Recorded times must increase")
    return phase, action, corners, tcp, grasping


def _record_pair_index(record, order, pair_count):
    """Resolve the goal pair, allowing legacy two-pick data without pair_index.

    Corrections may revisit either pair, so their record order cannot be used
    to infer the target pair. They must carry an explicit pair_index.
    """
    if "pair_index" in record:
        pair_index = record["pair_index"]
    elif order < pair_count:
        pair_index = order
    else:
        raise ValueError("Correction record is missing pair_index")

    if type(pair_index) is not int or not 0 <= pair_index < pair_count:
        raise ValueError(f"pair_index must be an integer from 0 to {pair_count - 1}")
    return pair_index


def load_episode(directory):
    """Read one successful episode, including valid correction pick/place samples."""
    folder = Path(directory)
    context = _read_json(folder, "context.json")
    metadata = _read_json(folder, "metadata.json")
    if metadata.get("success") is not True:
        raise ValueError("BC accepts successful episodes only")
    mode = context.get("grasp_mode")
    if mode not in MODES:
        raise ValueError("context.json must identify a valid grasp_mode")
    if context.get("place_xy") != "measured TCP immediately before opening":
        raise ValueError("Unsupported place_xy convention")
    expected_pick = ("measured TCP at manual close" if mode == "physical"
                     else "measured cloth corner at grasp")
    if context.get("pick_xy") != expected_pick:
        raise ValueError("Unsupported pick_xy convention")
    pairs = _pairs(context.get("pairs"))
    records = [json.loads(line) for line in (folder / "high_level.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Episode has no high-level transitions")

    high_obs, high_actions, low_obs, low_actions = [], [], [], []
    with np.load(folder / "low_level.npz", allow_pickle=False) as log:
        phases, commands, corners, tcp, grasping = _check_log(log)
        last_end = 0
        for order, record in enumerate(records):
            pair_index = _record_pair_index(record, order, len(pairs))
            start, end = record["low_level_start"], record["low_level_end"]
            if (type(start) is not int or type(end) is not int
                    or not (0 <= last_end <= start < end <= len(commands))):
                raise ValueError("Invalid, overlapping or out-of-order low-level segments")
            last_end = end

            high_obs.append(encode_observation(record["observation"]["corners"], pairs, mode))
            # Legacy virtual record stores the corner pick, not TCP XY; use the
            # first recorded TCP after grasp, consistent with the original loader.
            pick = (tcp[start, :2] if mode == "virtual" else record["action"]["pick"])
            high_actions.append(normalize_action(pick))

            for t in range(start + 1, end):
                phase = str(phases[t]).strip()
                prev_phase = str(phases[t - 1]).strip()
                allowed_phases = {CARRY_PHASE, *RELEASE_PHASES}

                if (
                    prev_phase not in allowed_phases
                    or phase not in allowed_phases
                    or (prev_phase in RELEASE_PHASES and phase == CARRY_PHASE)
                ):
                    continue

                prior_corners = corners[t - 1]
                pair_distances = np.array([
                    np.linalg.norm(prior_corners[a] - prior_corners[b])
                    for a, b in pairs
                ], dtype=np.float32)
                low_obs.append(encode_low_observation(
                    prior_corners, tcp[t - 1], commands[t - 1], pair_index,
                    bool(grasping[t - 1]), pair_distances,
                ))
                low_actions.append(np.asarray(commands[t], dtype=np.float32).copy())

    if not low_obs:
        raise ValueError("No valid carry/release transitions available for low-level BC")
    return Episode(
        path=str(folder.resolve()), mode=mode,
        observations=np.stack(high_obs).astype(np.float32),
        actions=np.stack(high_actions).astype(np.float32),
        low_observations=np.stack(low_obs).astype(np.float32),
        low_actions=np.stack(low_actions).astype(np.float32),
    )


def load_episodes(roots):
    """Return (valid episodes, rejected episodes with reasons)."""
    paths = sorted({p.parent.resolve() for root in roots
                    for p in Path(root).rglob("high_level.jsonl")})
    episodes, rejected = [], []
    for path in paths:
        try:
            episodes.append(load_episode(path))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    return episodes, rejected


def split_episodes(episodes, validation_fraction=.2, seed=0):
    """Use entire episodes (not individual frames) for train/validation splitting."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for train/validation splitting")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for mode in MODES:
        group = [ep for ep in episodes if ep.mode == mode]
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * validation_fraction))) if len(group) > 1 else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    if not validation:
        raise ValueError("Need two episodes of at least one grasp mode for validation")
    return train, validation
