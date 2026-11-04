"""Utilities for selecting and tracking four cloth corner simulation nodes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import torch


CORNER_LABELS = ("u_min_v_min", "u_min_v_max", "u_max_v_max", "u_max_v_min")


def _validate_nodes(nodes_w: torch.Tensor) -> None:
    if nodes_w.ndim != 2 or nodes_w.shape[1] != 3:
        raise ValueError(f"Expected node positions with shape (N, 3), got {tuple(nodes_w.shape)}")
    if nodes_w.shape[0] < 4:
        raise ValueError("At least four cloth nodes are required")
    if not torch.isfinite(nodes_w).all():
        raise ValueError("Cloth node positions contain NaN or Inf")


def _argmax_with_tolerance(values: torch.Tensor, tolerance: float) -> int:
    """Return the smallest index whose value is effectively the maximum."""
    maximum = values.max()
    candidates = torch.nonzero(values >= maximum - tolerance, as_tuple=False).flatten()
    return int(candidates.min().item())


def _order_corners_by_world_quadrant(nodes_w: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Order four corners by their world-space x/y quadrants.

    At symmetric boundary cases such as a square rotated by exactly 45 degrees,
    several assignments can be geometrically equivalent.  A small score
    tolerance and the vertex-index tuple provide a deterministic tie-break.
    """
    corners_xy = nodes_w.index_select(0, indices)[:, :2]
    centered_xy = corners_xy - corners_xy.mean(dim=0, keepdim=True)
    scale = centered_xy.abs().amax(dim=0).clamp_min(torch.finfo(nodes_w.dtype).eps)
    normalized_xy = centered_xy / scale
    targets = nodes_w.new_tensor(((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)))

    scored_permutations: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    index_values = tuple(int(value) for value in indices.detach().cpu().tolist())
    for order in permutations(range(4)):
        order_tensor = torch.tensor(order, dtype=torch.long, device=nodes_w.device)
        score = float(torch.sum((normalized_xy.index_select(0, order_tensor) - targets) ** 2).item())
        ordered_indices = tuple(index_values[position] for position in order)
        scored_permutations.append((score, ordered_indices, order))

    minimum_score = min(item[0] for item in scored_permutations)
    # At 45 degrees two world-quadrant assignments are theoretically tied.
    # Allow small simulation noise to remain inside that tie, then use vertex
    # indices below as the stable final decision.
    score_tolerance = 5.0e-2
    tied_assignments = [
        item for item in scored_permutations if item[0] <= minimum_score + score_tolerance
    ]
    _, _, best_order = min(tied_assignments, key=lambda item: item[1])
    order_tensor = torch.tensor(best_order, dtype=torch.long, device=nodes_w.device)
    return indices.index_select(0, order_tensor)


def _validate_square_geometry(nodes_w: torch.Tensor, indices: torch.Tensor) -> None:
    """Reject a four-point selection that is not approximately square."""
    corners = nodes_w.index_select(0, indices)
    edges = torch.linalg.vector_norm(corners.roll(-1, dims=0) - corners, dim=1)
    diagonals = torch.stack(
        (
            torch.linalg.vector_norm(corners[2] - corners[0]),
            torch.linalg.vector_norm(corners[3] - corners[1]),
        )
    )
    edge_mean = edges.mean()
    eps = torch.finfo(nodes_w.dtype).eps
    if float(edge_mean.item()) <= eps:
        raise RuntimeError("Corner detection produced a degenerate square with zero-length edges")

    edge_relative_error = torch.max(torch.abs(edges - edge_mean)) / edge_mean
    diagonal_ratio = diagonals / edge_mean
    expected_ratio = nodes_w.new_tensor(2.0).sqrt()
    diagonal_relative_error = torch.max(torch.abs(diagonal_ratio - expected_ratio)) / expected_ratio
    if float(edge_relative_error.item()) > 0.35 or float(diagonal_relative_error.item()) > 0.25:
        raise RuntimeError(
            "Detected corners do not form an approximate square "
            f"(edge error={float(edge_relative_error.item()):.3f}, "
            f"diagonal error={float(diagonal_relative_error.item()):.3f})"
        )


def detect_corner_vertex_indices(nodes_w: torch.Tensor) -> torch.Tensor:
    """
    Detect four cloth corner vertices without relying on PCA axis direction.

    1. Select a point farthest from the cloth center.
    2. Select the point farthest from the first point as its opposite.
    3. Select the point farthest from that diagonal, then its opposite corner.
    4. Order the result by world-space x/y quadrant for stable labels.
    """
    _validate_nodes(nodes_w)

    center = nodes_w.mean(dim=0)
    radius = torch.linalg.vector_norm(nodes_w - center, dim=1)
    tie_tolerance = max(float(radius.max().item()) * 5.0e-3, torch.finfo(nodes_w.dtype).eps)
    first = _argmax_with_tolerance(radius, tie_tolerance)

    distance_from_first = torch.linalg.vector_norm(nodes_w - nodes_w[first], dim=1)
    opposite = int(torch.argmax(distance_from_first).item())

    remaining_mask = torch.ones(nodes_w.shape[0], dtype=torch.bool, device=nodes_w.device)
    remaining_mask[first] = False
    remaining_mask[opposite] = False
    remaining_indices = torch.nonzero(remaining_mask, as_tuple=False).flatten()

    diagonal = nodes_w[opposite] - nodes_w[first]
    diagonal_length_squared = torch.dot(diagonal, diagonal)
    if float(diagonal_length_squared.item()) <= torch.finfo(nodes_w.dtype).eps:
        raise RuntimeError("Corner detection could not form a non-zero diagonal")
    offsets = nodes_w.index_select(0, remaining_indices) - nodes_w[first]
    projection = (offsets @ diagonal)[:, None] * diagonal[None, :] / diagonal_length_squared
    distance_from_diagonal = torch.linalg.vector_norm(offsets - projection, dim=1)
    third_position = _argmax_with_tolerance(distance_from_diagonal, tie_tolerance)
    third = int(remaining_indices[third_position].item())

    fourth_candidates = remaining_indices[remaining_indices != third]
    distance_from_third = torch.linalg.vector_norm(
        nodes_w.index_select(0, fourth_candidates) - nodes_w[third], dim=1
    )
    fourth = int(fourth_candidates[torch.argmax(distance_from_third)].item())
    indices = torch.tensor((first, third, opposite, fourth), dtype=torch.long, device=nodes_w.device)

    if torch.unique(indices).numel() != 4:
        raise RuntimeError(
            "Corner detection selected duplicate nodes. The cloth may be folded, degenerate, or not initialized."
        )
    indices = _order_corners_by_world_quadrant(nodes_w, indices)
    _validate_square_geometry(nodes_w, indices)
    return indices


def get_corner_world_positions(nodes_w: torch.Tensor, corner_indices: torch.Tensor) -> torch.Tensor:
    """Read the current world positions of the four previously selected nodes"""
    _validate_nodes(nodes_w)
    if corner_indices.shape != (4,):
        raise ValueError(f"Expected four corner indices, got shape {tuple(corner_indices.shape)}")
    if corner_indices.device != nodes_w.device:
        corner_indices = corner_indices.to(nodes_w.device)
    if int(corner_indices.min().item()) < 0 or int(corner_indices.max().item()) >= nodes_w.shape[0]:
        raise IndexError("A corner index is outside the current cloth node array")
    return nodes_w.index_select(0, corner_indices)


@dataclass
class CornerMotionVerifier:
    """Accumulate the maximum displacement observed for each tracked corner."""

    initial_positions: torch.Tensor
    max_displacements: torch.Tensor

    @classmethod
    def create(cls, initial_positions: torch.Tensor) -> "CornerMotionVerifier":
        if initial_positions.shape != (4, 3):
            raise ValueError("Expected initial corner positions with shape (4, 3)")
        initial = initial_positions.detach().clone()
        return cls(initial_positions=initial, max_displacements=initial.new_zeros(4))

    def update(self, current_positions: torch.Tensor) -> torch.Tensor:
        displacement = torch.linalg.vector_norm(current_positions - self.initial_positions, dim=1)
        self.max_displacements = torch.maximum(self.max_displacements, displacement)
        return displacement

    def moved_mask(self, threshold_m: float) -> torch.Tensor:
        return self.max_displacements >= threshold_m

