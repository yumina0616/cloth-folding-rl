"""Pinch guide checks without Isaac Sim: python -m pytest tests/test_teleop_pinch_guide.py"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/grasp"))
from teleop_pinch_guide import GREEN, RED, marker_color, next_source_slot, pair_with_source, xy_distance  # noqa: E402

CORNERS = torch.tensor([[0.35, -0.20], [0.65, -0.20], [0.65, 0.10], [0.35, 0.10]])
PAIRS = [(3, 2), (0, 1)]  # near-left -> far-left, then near-right -> far-right


def test_first_source_corner_comes_first():
    assert next_source_slot(PAIRS, CORNERS, CORNERS.clone()) == 3


def test_carried_corner_moves_on_to_the_next_pair():
    now = CORNERS.clone()
    now[3] = CORNERS[2]                       # first source corner was carried onto its destination
    assert next_source_slot(PAIRS, CORNERS, now) == 0


def test_dragged_second_corner_is_not_skipped():
    now = CORNERS.clone()
    now[3] = CORNERS[2]                       # first pair folded
    now[0] += torch.tensor([0.06, 0.03])      # second source dragged 6.7 cm while folding the first
    assert next_source_slot(PAIRS, CORNERS, now) == 0


def test_failed_grasp_does_not_skip_a_corner():
    now = CORNERS.clone()
    now[3] += torch.tensor([0.01, 0.0])       # closed and opened without lifting: moved only 1 cm
    assert next_source_slot(PAIRS, CORNERS, now) == 3


def test_none_after_every_pair():
    now = CORNERS.clone()
    now[3], now[0] = CORNERS[2], CORNERS[1]
    assert next_source_slot(PAIRS, CORNERS, now) is None


def test_color_uses_horizontal_distance_only():
    point = torch.tensor([0.40, 0.05, 0.0015])
    above = torch.tensor([0.403, 0.05, 0.08])  # 3 mm off in x, 8 cm above
    assert abs(xy_distance(above, point) - 0.003) < 1e-6
    assert marker_color(xy_distance(above, point), 0.005) == GREEN
    assert marker_color(0.006, 0.005) == RED


def test_pair_with_source_gives_the_destination_or_none():
    assert pair_with_source(PAIRS, 0) == (0, 1)
    assert pair_with_source(PAIRS, 2) is None   # a destination corner is never a source
