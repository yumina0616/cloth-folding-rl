"""PairDistanceLogger writes raw per-pair distances; run: python -m pytest tests/test_pair_distance_logger.py"""
import csv

import torch

from scripts.rl.pair_distance_logger import PairDistanceLogger

# 0 ─── 1
# │     │
# 2 ─── 3
FLAT = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]])
PAIRS = [(0, 1), (2, 3)]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_one_row_per_step_and_env_with_each_pair(tmp_path):
    path = tmp_path / "pairs.csv"
    logger = PairDistanceLogger(path, PAIRS)

    moved = FLAT.clone()
    moved[0, 0] = 0.15  # env 1: pair 0-1 closes to 0.05 m, pair 2-3 stays 0.2 m
    logger.record(0, torch.stack([FLAT, FLAT]))
    logger.record(1, torch.stack([FLAT, moved]))

    rows = read_rows(path)
    assert list(rows[0]) == ["step", "env_id", "pair_0_1", "pair_2_3"]
    assert [(r["step"], r["env_id"]) for r in rows] == [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")]
    assert float(rows[3]["pair_0_1"]) == 0.05
    assert float(rows[3]["pair_2_3"]) == 0.2


def test_new_logger_starts_a_fresh_file(tmp_path):
    path = tmp_path / "pairs.csv"
    PairDistanceLogger(path, PAIRS).record(0, FLAT[None])
    PairDistanceLogger(path, PAIRS)  # same path again -> old rows gone
    assert read_rows(path) == []
