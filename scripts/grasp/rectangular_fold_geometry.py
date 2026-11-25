"""Material-corner pairing and released-cloth success metrics."""
import math

import torch


def rectangular_pairs(corners, robot_xy):
    """Choose the adjacent edge nearest the robot and pair across the sheet.

    Returns indices into the tracker's persistent four-corner array, never
    diagonal pairs. Sorting is used only at initialization, before folding.
    """
    center = corners[:, :2].mean(0)
    delta = corners[:, :2] - center
    order = torch.atan2(delta[:, 1], delta[:, 0]).argsort().tolist()
    edges = [(order[i], order[(i + 1) % 4]) for i in range(4)]
    distances = [float(torch.linalg.vector_norm((corners[a, :2] + corners[b, :2]) / 2 - robot_xy))
                 for a, b in edges]
    i = min(range(4), key=distances.__getitem__)
    return [(order[i], order[(i - 1) % 4]), (order[(i + 1) % 4], order[(i + 2) % 4])]


def pair_distances(corners, pairs):
    return [float(torch.linalg.vector_norm(corners[a] - corners[b])) for a, b in pairs]


def cloth_yaw_from_corners(corners_xy):
    """Rotation of a square cloth about world z [rad], from its four corners (any order).

    A square looks the same every 90 deg, so the answer is the edge direction closest to
    world x, in [-pi/4, pi/4). 0 means the edges are parallel to world x and y.
    """
    xy = torch.as_tensor(corners_xy, dtype=torch.float64)[:, :2]
    if xy.shape != (4, 2):
        raise ValueError("corners_xy must have shape (4, 2) or (4, 3)")
    delta = xy - xy.mean(0)
    order = torch.atan2(delta[:, 1], delta[:, 0]).argsort()
    ring = xy[order]
    edges = torch.roll(ring, -1, dims=0) - ring
    angles = torch.atan2(edges[:, 1], edges[:, 0])
    # Average the four edge directions modulo 90 deg (circular mean of 4 * angle)
    four = torch.atan2(torch.sin(4 * angles).sum(), torch.cos(4 * angles).sum())
    yaw = float(four) / 4
    return yaw - math.pi / 2 if yaw >= math.pi / 4 else yaw


def to_cloth_frame(points_xy, center_xy, yaw):
    """World xy -> cloth axes (x along the cloth edge nearest world x), origin at center_xy."""
    d = torch.as_tensor(points_xy)[..., :2] - torch.as_tensor(center_xy)[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.stack((d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c), dim=-1)


def axis_pairs(corners, robot_xy, axis, yaw=0.0):
    """Pairs that move the two corners nearest the robot along a cloth axis ("x" or "y").

    With yaw = 0 this is the old --move_axis x/y rule in world axes; with a rotated cloth the
    axes are the cloth's own edges.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    xy = torch.as_tensor(corners)[:, :2]
    center = xy.mean(0)
    c = to_cloth_frame(xy, center, yaw)
    robot = to_cloth_frame(torch.as_tensor(robot_xy)[:2], center, yaw)
    k = 0 if axis == "x" else 1
    side = 1.0 if float(robot[k]) >= 0 else -1.0
    order = sorted(range(4), key=lambda i: -side * float(c[i, k]))
    dests = order[2:]
    return [(src, min(dests, key=lambda d: abs(float(c[d, 1 - k] - c[src, 1 - k])))) for src in order[:2]]
