"""Hand goals for one pick-and-place; run without Isaac Sim: python -m pytest tests/test_fold_motion.py"""
import math

import torch

from scripts.rl.fold_motion import (
    ARC_SEGMENTS, carry_arc_goals, lower_goal, place_point_from_xy, placement_clearance, smoothstep_path,
)

TOL = 1e-6
THICK = 0.003


def test_smoothstep_path_shape_and_end_points():
    start = torch.zeros(2, 3)
    goal = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    path = smoothstep_path(start, goal, 10)
    assert path.shape == (10, 2, 3)
    assert torch.allclose(path[-1], goal)
    # Eased: the first step is much smaller than an even 1/10
    assert float(path[0, 0, 0]) < 0.1 / 2


def test_smoothstep_path_keeps_a_waiting_environment_still():
    start = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    goal = torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.5, 0.5]])  # env 0 is not moving
    path = smoothstep_path(start, goal, 5)
    assert torch.allclose(path[:, 0], start[0].expand(5, 3))


def test_place_point_sits_half_a_thickness_above_the_table():
    point = place_point_from_xy(torch.tensor([[0.4, -0.1]]), table_z=0.0, cloth_thickness=THICK)
    assert torch.allclose(point, torch.tensor([[0.4, -0.1, THICK / 2]]))


def test_carry_arc_starts_up_and_ends_above_place_point():
    start = torch.tensor([[0.35, -0.20, 0.0015]])
    place = torch.tensor([[0.35, 0.10, 0.0015]])     # 0.30 m away
    offset = torch.tensor([[0.0, 0.0, -0.015]])      # corner 1.5 cm below the TCP
    goals = carry_arc_goals(start, place, offset, lift_height=0.12, cloth_thickness=THICK)
    assert goals.shape == (ARC_SEGMENTS, 1, 3)
    corner_path = goals + offset                     # where the held corner goes
    # Arc height = max(lift_height, span / 2) = 0.15 m, reached at the middle segment
    middle = corner_path[ARC_SEGMENTS // 2 - 1, 0]
    assert abs(float(middle[2]) - (0.0015 + 0.15 + placement_clearance(THICK) * 0.5)) < 1e-4
    # Last goal: held corner right above the place point by the clearance
    assert torch.allclose(corner_path[-1, 0, :2], place[0, :2], atol=TOL)
    assert abs(float(corner_path[-1, 0, 2]) - (0.0015 + placement_clearance(THICK))) < TOL


def test_short_carry_uses_lift_height():
    start = torch.zeros(1, 3)
    place = torch.tensor([[0.05, 0.0, 0.0]])         # span / 2 = 0.025 < lift_height
    goals = carry_arc_goals(start, place, torch.zeros(1, 3), lift_height=0.12, cloth_thickness=THICK)
    top = float(goals[:, 0, 2].max())
    assert abs(top - (0.12 + placement_clearance(THICK) * 0.5)) < 1e-4


def test_each_environment_gets_its_own_arc():
    start = torch.zeros(2, 3)
    place = torch.tensor([[0.30, 0.0, 0.0], [0.0, 0.40, 0.0]])
    goals = carry_arc_goals(start, place, torch.zeros(2, 3), lift_height=0.12, cloth_thickness=THICK)
    assert torch.allclose(goals[-1, :, :2], place[:, :2], atol=TOL)
    # Longer span -> higher arc (0.20 vs 0.15)
    assert float(goals[:, 1, 2].max()) > float(goals[:, 0, 2].max())


def test_lower_goal_matches_last_arc_point():
    start = torch.tensor([[0.3, -0.2, 0.0015]])
    place = torch.tensor([[0.3, 0.1, 0.0015]])
    offset = torch.tensor([[0.01, 0.0, -0.015]])
    arc = carry_arc_goals(start, place, offset, lift_height=0.12, cloth_thickness=THICK)
    assert torch.allclose(lower_goal(place, offset, THICK), arc[-1], atol=TOL)


def test_inputs_are_not_modified():
    start = torch.tensor([[0.3, -0.2, 0.0015]])
    place = torch.tensor([[0.3, 0.1, 0.0015]])
    offset = torch.tensor([[0.0, 0.0, -0.015]])
    before = (start.clone(), place.clone(), offset.clone())
    carry_arc_goals(start, place, offset, lift_height=0.12, cloth_thickness=THICK)
    lower_goal(place, offset, THICK)
    assert all(torch.equal(a, b) for a, b in zip(before, (start, place, offset)))
    assert math.isclose(placement_clearance(0.001), 0.003)
