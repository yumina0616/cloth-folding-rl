"""Material-corner pairing and released-cloth success metrics."""
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
