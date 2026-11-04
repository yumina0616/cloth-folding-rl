"""Synthetic tests for rotation-invariant cloth corner detection."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "corner_tracking" / "cloth_corner_tracker.py"
SPEC = importlib.util.spec_from_file_location("cloth_corner_tracker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CornerRotationTest(unittest.TestCase):
    def _make_grid(
        self, yaw_degrees: float, noise_m: float, seed: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        coordinates = torch.linspace(-0.2, 0.2, 40, dtype=dtype)
        xx, yy = torch.meshgrid(coordinates, coordinates, indexing="ij")
        nodes = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(1600, dtype=dtype)), dim=1)

        yaw = math.radians(yaw_degrees)
        rotation = nodes.new_tensor(
            ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
        )
        nodes[:, :2] = nodes[:, :2] @ rotation.T
        if noise_m:
            generator = torch.Generator().manual_seed(seed)
            nodes += torch.randn(nodes.shape, dtype=nodes.dtype, generator=generator) * noise_m
        return nodes

    def test_rotations_and_noise_select_actual_corners_with_stable_labels(self) -> None:
        expected = torch.tensor((0, 39, 1599, 1560), dtype=torch.long)
        for dtype in (torch.float32, torch.float64):
            for yaw_degrees in (0.0, 10.0, 30.0, 45.0):
                for noise_m in (0.0, 0.0001):
                    with self.subTest(dtype=dtype, yaw_degrees=yaw_degrees, noise_m=noise_m):
                        for seed in range(8):
                            nodes = self._make_grid(yaw_degrees, noise_m, seed, dtype)
                            actual = MODULE.detect_corner_vertex_indices(nodes).cpu()
                            self.assertTrue(
                                torch.equal(actual, expected), (dtype, yaw_degrees, noise_m, seed, actual)
                            )

    def test_corner_position_api_is_unchanged(self) -> None:
        nodes = self._make_grid(30.0, 0.0001, seed=0)
        indices = MODULE.detect_corner_vertex_indices(nodes)
        positions = MODULE.get_corner_world_positions(nodes, indices)
        self.assertEqual(tuple(positions.shape), (4, 3))
        self.assertTrue(torch.equal(positions, nodes.index_select(0, indices)))

    def test_shuffled_translated_nodes_select_true_corners(self) -> None:
        # Mesh vertex ordering and placement must not determine corner selection.
        true_corners = torch.tensor((0, 39, 1599, 1560))
        for yaw in (-45.0, -30.0, 60.0, 90.0, 135.0, 180.0, 270.0):
            for seed in range(8):
                with self.subTest(yaw=yaw, seed=seed):
                    nodes = self._make_grid(yaw, 0.0001, seed)
                    permutation = torch.randperm(1600, generator=torch.Generator().manual_seed(seed))
                    shuffled = nodes[permutation] + nodes.new_tensor((0.5, -0.3, 0.012))
                    indices = MODULE.detect_corner_vertex_indices(shuffled)
                    original_indices = permutation[indices]
                    self.assertEqual(set(original_indices.tolist()), set(true_corners.tolist()))

    def test_invalid_geometry_is_rejected(self) -> None:
        degenerate = torch.zeros((4, 3))
        rectangle = torch.tensor(((-1., -0.1, 0.), (-1., 0.1, 0.),
                                  (1., 0.1, 0.), (1., -0.1, 0.)))
        for nodes in (degenerate, rectangle):
            with self.subTest(nodes=nodes):
                with self.assertRaises(RuntimeError):
                    MODULE.detect_corner_vertex_indices(nodes)


if __name__ == "__main__":
    unittest.main()
