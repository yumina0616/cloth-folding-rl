"""Utilities for selecting and tracking four cloth corner simulation nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


CORNER_LABELS = ("u_min_v_min", "u_min_v_max", "u_max_v_max", "u_max_v_min")


def _validate_nodes(nodes_w: torch.Tensor) -> None:
    if nodes_w.ndim != 2 or nodes_w.shape[1] != 3:
        raise ValueError(f"Expected node positions with shape (N, 3), got {tuple(nodes_w.shape)}")
    if nodes_w.shape[0] < 4:
        raise ValueError("At least four cloth nodes are required")
    if not torch.isfinite(nodes_w).all():
        raise ValueError("Cloth node positions contain NaN or Inf")


def detect_corner_vertex_indices(nodes_w: torch.Tensor) -> torch.Tensor:
    """
    Detect four cloth corner vertices using PCA.

    1. Project 3D cloth nodes onto the two principal axes.
    2. Normalize the projected coordinates to [0, 1].
    3. Find the node closest to each of the four 2D corners.
    """
    _validate_nodes(nodes_w)

    centered = nodes_w - nodes_w.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(nodes_w.shape[0] - 1, 1)  # PCA
    _, eigenvectors = torch.linalg.eigh(covariance)
    plane_basis = eigenvectors[:, -2:]
    uv = centered @ plane_basis

    uv_min = uv.amin(dim=0)
    uv_max = uv.amax(dim=0)
    span = (uv_max - uv_min).clamp_min(torch.finfo(nodes_w.dtype).eps)
    uv_normalized = (uv - uv_min) / span
    targets = nodes_w.new_tensor(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))
    squared_distance = torch.sum((uv_normalized[:, None, :] - targets[None, :, :]) ** 2, dim=-1)
    indices = torch.argmin(squared_distance, dim=0)

    if torch.unique(indices).numel() != 4:
        raise RuntimeError(
            "Corner detection selected duplicate nodes. The cloth may be folded, degenerate, or not initialized."
        )
    return indices.to(dtype=torch.long)


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

