"""Batched hand goals for one pick-and-place, as pure tensor math (no Isaac Sim).

Same motion as the virtual fold demo (scripts/grasp/rectangular_fold_demo.py, place_corner):
approach 8 cm above, descend to 1.5 cm above the corner, carry along a raised arc, lower, release.
The only difference: the policy gives WHERE the held corner should land (place_xy), instead of
"onto corner number k". The demo follows the destination corner while it moves; here the
place point is fixed when the decision is made, so the policy has to anticipate that motion.

Shapes: N = number of environments. Positions are world coordinates in m.
"""
from __future__ import annotations

import math

import torch

APPROACH_HEIGHT = 0.08  # above the corner before descending (demo: above[2] += 0.08)
TOUCH_HEIGHT = 0.015    # TCP stops this far above the corner when grasping virtually (demo: touch[2] += 0.015)
RETREAT_HEIGHT = 0.10   # straight up after release (demo: retreat[2] += 0.10)
ARC_SEGMENTS = 8        # demo: for segment in range(1, 9)


def smoothstep_path(start, goal, count):
    """Hand targets from start to goal, eased at both ends. (N, 3), (N, 3) -> (count, N, 3).

    Same easing as the demo's move(): alpha = t^2 (3 - 2t), the last row equals goal.
    """
    t = torch.arange(1, count + 1, dtype=start.dtype, device=start.device) / count
    alpha = (t * t * (3 - 2 * t)).view(count, 1, 1)
    return start.unsqueeze(0) + alpha * (goal - start).unsqueeze(0)


def placement_clearance(cloth_thickness):
    """How far above the destination the held corner is put down (demo: max(0.003, thickness + 0.001))."""
    return max(0.003, cloth_thickness + 0.001)


def carry_arc_goals(start_corner, place_point, offset, lift_height, cloth_thickness, segments=ARC_SEGMENTS):
    """TCP goals that carry the held corner along a raised arc to place_point. -> (segments, N, 3).

    start_corner: (N, 3) held corner when the carry starts
    place_point:  (N, 3) where the held corner should land (z = resting height of the corner)
    offset:       (N, 3) held corner - TCP, measured right after the grasp (virtual grasp keeps it fixed)
    Arc height = max(lift_height, half the horizontal span), as in the demo (not the low correction arc).
    """
    span = torch.linalg.vector_norm(place_point[:, :2] - start_corner[:, :2], dim=-1)  # (N,)
    arc_height = torch.clamp(span / 2, min=lift_height)                                 # (N,)
    clearance = placement_clearance(cloth_thickness)
    goals = []
    for segment in range(1, segments + 1):
        t = segment / segments
        alpha = (1 - math.cos(math.pi * t)) / 2
        node_goal = torch.lerp(start_corner, place_point, alpha)
        node_goal[:, 2] += arc_height * math.sin(math.pi * t) + clearance * alpha
        goals.append(node_goal - offset)
    return torch.stack(goals)


def lower_goal(place_point, offset, cloth_thickness):
    """TCP goal that puts the held corner just above place_point. -> (N, 3)."""
    goal = place_point - offset
    goal[:, 2] += placement_clearance(cloth_thickness)
    return goal


def place_point_from_xy(place_xy, table_z, cloth_thickness):
    """(N, 2) place point from the policy -> (N, 3), at the resting height of a corner on the table.

    A flat corner node sits about half a cloth thickness above the table top.
    """
    z = torch.full_like(place_xy[:, :1], table_z + cloth_thickness / 2)
    return torch.cat([place_xy, z], dim=-1)
