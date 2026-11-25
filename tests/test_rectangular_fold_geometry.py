"""Cloth rotation helpers for the physical fold; runs without Isaac Sim."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

GRASP = Path(__file__).parents[1] / "scripts" / "grasp"


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", GRASP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEOMETRY = load("rectangular_fold_geometry")
PICKER = load("physical_picker")


def square(yaw_deg, center=(0.5, -0.05), half=0.15):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    return torch.tensor([[center[0] + x * c - y * s, center[1] + x * s + y * c, 0.0] for x, y in local],
                        dtype=torch.float64)


class ClothYawTest(unittest.TestCase):
    def test_measured_yaw(self):
        for yaw in (0.0, 10.0, -15.0, 30.0, 44.0):
            self.assertAlmostEqual(math.degrees(GEOMETRY.cloth_yaw_from_corners(square(yaw))), yaw, places=6)
        # A square repeats every 90 deg: 60 deg looks like -30 deg
        self.assertAlmostEqual(math.degrees(GEOMETRY.cloth_yaw_from_corners(square(60.0))), -30.0, places=6)

    def test_axis_pairs_unchanged_without_rotation(self):
        corners = square(0.0)
        robot = torch.tensor([0.0, 0.0])
        for axis in ("x", "y"):
            c = corners[:, :2]
            center = c.mean(0)
            k = 0 if axis == "x" else 1
            side = 1.0 if float(robot[k] - center[k]) >= 0 else -1.0
            order = sorted(range(4), key=lambda i: -side * float(c[i, k] - center[k]))
            old = [(src, min(order[2:], key=lambda d: abs(float(c[d, 1 - k] - c[src, 1 - k])))) for src in order[:2]]
            self.assertEqual(GEOMETRY.axis_pairs(corners, robot, axis), old)

    def test_axis_pairs_follow_the_cloth(self):
        robot = torch.tensor([0.0, 0.0])
        for axis in ("x", "y"):
            expected = GEOMETRY.axis_pairs(square(0.0), robot, axis)
            self.assertEqual(GEOMETRY.axis_pairs(square(15.0), robot, axis, math.radians(15.0)), expected)


class RotatedPinchTest(unittest.TestCase):
    def test_pinch_point_rotates_with_the_cloth(self):
        base = square(0.0)
        for yaw_deg in (0.0, 15.0, -20.0):
            yaw = math.radians(yaw_deg)
            rotated = square(yaw_deg)
            center = rotated[:, :2].mean(0)
            for i in range(4):
                got = PICKER.physical_grasp_xy(rotated[i, :2], center, 0.04, jaw_shift=0.01, axes_yaw=yaw)
                plain = PICKER.physical_grasp_xy(base[i, :2], base[:, :2].mean(0), 0.04, jaw_shift=0.01)
                # Offset from the corner, expressed in the cloth axes, is the same as without rotation
                local = GEOMETRY.to_cloth_frame(got, rotated[i, :2], yaw)
                self.assertTrue(torch.allclose(local, plain - base[i, :2], atol=1e-9), (yaw_deg, i))


class RotateNodesTest(unittest.TestCase):
    def test_rigid_rotation_keeps_shape_and_sets_yaw(self):
        flat = square(0.0)
        rotated = GEOMETRY.rotate_nodes_about_center(flat, math.radians(40.0))
        self.assertTrue(torch.allclose(rotated[:, :2].mean(0), flat[:, :2].mean(0)))
        self.assertTrue(torch.allclose(torch.cdist(rotated, rotated), torch.cdist(flat, flat)))
        self.assertAlmostEqual(math.degrees(GEOMETRY.cloth_yaw_from_corners(rotated)), 40.0, places=6)


def grid(yaw_deg, n=33, size=0.30, center=(0.5, -0.05)):
    g = torch.linspace(-size / 2, size / 2, n, dtype=torch.float64)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    flat = torch.stack((X.ravel() + center[0], Y.ravel() + center[1], torch.zeros(n * n, dtype=torch.float64)), 1)
    return GEOMETRY.rotate_nodes_about_center(flat, math.radians(yaw_deg))


class LayerYawTest(unittest.TestCase):
    def test_edge_points_lie_along_the_two_edges(self):
        for yaw_deg in (0.0, 20.0):
            nodes = grid(yaw_deg)
            corner = 0  # first grid node is a corner
            yaw = math.radians(yaw_deg)
            ex, ey = GEOMETRY.edge_reference_vertices(nodes[:, :2], corner, nodes[:, :2].mean(0), yaw)
            lx = GEOMETRY.to_cloth_frame(nodes[ex, :2], nodes[corner, :2], yaw)
            ly = GEOMETRY.to_cloth_frame(nodes[ey, :2], nodes[corner, :2], yaw)
            self.assertAlmostEqual(float(lx[1]), 0.0, places=6)   # on the x edge
            self.assertAlmostEqual(float(ly[0]), 0.0, places=6)   # on the y edge
            self.assertAlmostEqual(abs(float(lx[0])), 0.05625, places=4)  # nearest node to 6 cm (grid 9.375 mm)

    def test_hand_yaw_follows_the_layer_and_ignores_flips(self):
        for yaw_deg in (0.0, 15.0, -30.0):
            nodes = grid(yaw_deg)
            yaw = math.radians(yaw_deg)
            ex, ey = GEOMETRY.edge_reference_vertices(nodes[:, :2], 0, nodes[:, :2].mean(0), yaw)
            got = GEOMETRY.layer_jaw_yaw(nodes[0, :2], nodes[ex, :2], nodes[ey, :2])
            self.assertAlmostEqual(math.degrees(got), yaw_deg, places=6)
            # Mirror the layer across its x edge (as a fold does): same jaw direction
            mirrored_y = 2 * nodes[0, :2] - nodes[ey, :2]
            got = GEOMETRY.layer_jaw_yaw(nodes[0, :2], nodes[ex, :2], mirrored_y)
            self.assertAlmostEqual(math.degrees(got), yaw_deg, places=6)


if __name__ == "__main__":
    unittest.main()
