"""Physical picker contract checks; run without launching Isaac Sim."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "grasp" / "physical_picker.py"
SPEC = importlib.util.spec_from_file_location("physical_picker_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PhysicalPicker = MODULE.PhysicalPicker
physical_grasp_xy = MODULE.physical_grasp_xy
pinch_vertex = MODULE.pinch_vertex
create_picker = MODULE.create_picker


class Cloth:
    def __init__(self):
        # one env, two vertices: a corner at the origin and another point
        self.data = SimpleNamespace(nodal_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]))


class GraspPointTest(unittest.TestCase):
    def test_axis_layout_matches_008(self):
        # 40 mm open: x = pad 10 + margin 5 = 15 mm, y = 20 + 15 = 35 mm toward the center
        xy = physical_grasp_xy(torch.tensor([0.0, 0.0]), torch.tensor([0.15, 0.15]), 0.04)
        self.assertTrue(torch.allclose(xy, torch.tensor([0.015, 0.035])))

    def test_jaw_along_x_swaps_the_room(self):
        # Jaw along the fold direction (world x): x = 20 + 15 = 35 mm, y = pad 15 mm
        xy = physical_grasp_xy(torch.tensor([0.0, 0.0]), torch.tensor([0.15, 0.15]), 0.04, jaw_axis="x")
        self.assertTrue(torch.allclose(xy, torch.tensor([0.035, 0.015])))
        with self.assertRaises(ValueError):
            physical_grasp_xy(torch.tensor([0.0, 0.0]), torch.tensor([0.15, 0.15]), 0.04, jaw_axis="z")

    def test_other_corner_points_inward(self):
        xy = physical_grasp_xy(torch.tensor([0.3, 0.3]), torch.tensor([0.15, 0.15]), 0.04)
        self.assertTrue(torch.allclose(xy, torch.tensor([0.285, 0.265])))

    def test_rejects_bad_width(self):
        with self.assertRaises(ValueError):
            physical_grasp_xy(torch.tensor([0.0, 0.0]), torch.tensor([0.15, 0.15]), 0.0)


class PinchVertexTest(unittest.TestCase):
    def setUp(self):
        # A flat 0.30 m cloth with 5 mm spacing; vertex 0 is the corner at the origin
        axis = torch.arange(0, 61) * 0.005
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        self.flat = torch.stack([gx.flatten(), gy.flatten(), torch.full_like(gx.flatten(), 0.002)], dim=1)
        self.center = torch.tensor([0.15, 0.15])
        self.vertex = pinch_vertex(self.flat[:, :2], 0, self.center, 0.04, jaw_axis="x")

    def test_flat_state_matches_the_grasp_point(self):
        # jaw along x: 35 mm along x and 15 mm along y inside the corner
        self.assertTrue(torch.allclose(self.flat[self.vertex, :2], torch.tensor([0.035, 0.015]), atol=1e-6))

    def test_stays_just_inside_the_corner_after_the_cloth_is_flipped(self):
        # Fold the whole cloth over about x = 0.15: the corner moves from x = 0 to x = 0.30
        flipped = self.flat.clone()
        flipped[:, 0] = 0.30 - flipped[:, 0]
        flipped[:, 2] = 0.008
        corner, pinch = flipped[0, :2], flipped[self.vertex, :2]
        # The pinch point is 35 mm toward the cloth (-x), 15 mm toward +y; a recomputed
        # "toward the original centre" direction would have pointed outside the cloth (+x)
        self.assertTrue(torch.allclose(pinch - corner, torch.tensor([-0.035, 0.015]), atol=1e-6))

    def test_stays_just_inside_the_corner_after_the_cloth_is_rotated(self):
        # Rotate the cloth 90 degrees about its centre: the pinch vertex must still be 38 mm from the corner
        offset = self.flat[:, :2] - self.center
        rotated = torch.stack([-offset[:, 1], offset[:, 0]], dim=1) + self.center
        gap = torch.linalg.vector_norm(rotated[self.vertex] - rotated[0])
        self.assertAlmostEqual(float(gap), (0.035 ** 2 + 0.015 ** 2) ** 0.5, places=5)

    def test_rejects_bad_vertex(self):
        with self.assertRaises(ValueError):
            pinch_vertex(self.flat[:, :2], 10 ** 6, self.center, 0.04)


class PhysicalPickerTest(unittest.TestCase):
    def setUp(self):
        self.cloth = Cloth()
        self.anchor = torch.tensor([0.015, 0.035, 0.01])
        self.picker = PhysicalPicker(self.cloth, lambda: self.anchor)

    def test_grasp_records_corner_without_writing(self):
        before = self.cloth.data.nodal_pos_w.clone()
        self.assertTrue(self.picker.grasp(torch.zeros(3), vertex_id=0))
        self.assertTrue(self.picker.is_grasping)
        self.assertTrue(torch.equal(self.cloth.data.nodal_pos_w, before))  # nothing is attached
        self.assertAlmostEqual(self.picker.drift(), 0.0)

    def test_rejects_far_gripper(self):
        self.anchor = torch.tensor([0.5, 0.5, 0.5])
        self.assertFalse(self.picker.grasp(torch.zeros(3), vertex_id=0))
        self.assertFalse(self.picker.is_grasping)

    def test_drift_and_live_target(self):
        self.picker.grasp(torch.zeros(3), vertex_id=0)
        self.anchor = self.anchor + torch.tensor([0.0, 0.0, 0.05])  # hand lifts 5 cm
        self.cloth.data.nodal_pos_w[0, 0, 2] = 0.03                  # corner follows only 3 cm
        self.assertAlmostEqual(self.picker.drift(), 0.02, places=6)
        self.assertTrue(torch.allclose(self.picker.target_position_w, torch.tensor([0.0, 0.0, 0.03])))

    def test_release(self):
        self.picker.grasp(torch.zeros(3), vertex_id=0)
        self.picker.release()
        self.assertFalse(self.picker.is_grasping)
        self.assertIsNone(self.picker.target_position_w)

    def test_create_picker(self):
        self.assertIsInstance(create_picker(self.cloth, lambda: self.anchor, grasp_mode="physical"), PhysicalPicker)
        with self.assertRaises(ValueError):
            create_picker(self.cloth, lambda: self.anchor, grasp_mode="magnet")


if __name__ == "__main__":
    unittest.main()
