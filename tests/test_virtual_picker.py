"""Attachment contract checks; run without launching Isaac Sim."""
import unittest
from types import SimpleNamespace

import torch

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "grasp" / "virtual_picker.py"
SPEC = importlib.util.spec_from_file_location("virtual_picker_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
VirtualPicker = MODULE.VirtualPicker
create_picker = MODULE.create_picker


class Cloth:
    def __init__(self, kinematic=False):
        positions = torch.tensor([[[0., 0., 0.], [0.1, 0., 0.]],
                                  [[1., 0., 0.], [1.1, 0., 0.]]])
        targets = torch.cat((positions.clone(), torch.ones(2, 2, 1)), dim=-1) if kinematic else None
        self.data = SimpleNamespace(nodal_pos_w=positions, nodal_vel_w=torch.zeros_like(positions),
                                    nodal_kinematic_target=targets)
        self.writes = 0

    def write_nodal_pos_to_sim_index(self, values, env_ids):
        self.data.nodal_pos_w[env_ids] = values
        self.writes += 1

    def write_nodal_velocity_to_sim_index(self, values, env_ids):
        self.data.nodal_vel_w[env_ids] = values
        self.writes += 1

    def write_nodal_kinematic_target_to_sim_index(self, values, env_ids):
        self.data.nodal_kinematic_target[env_ids] = values
        self.writes += 1


class PickerTests(unittest.TestCase):
    def test_surface_tracks_measured_anchor_with_offset_and_velocity(self):
        cloth = Cloth()
        anchor = torch.tensor([0., 0., 0.01])
        picker = VirtualPicker(cloth, lambda: anchor)
        self.assertTrue(picker.grasp([0., 0., 0.]))
        anchor += torch.tensor([0., 0.02, 0.05])
        picker.update(dt=0.01)
        torch.testing.assert_close(cloth.data.nodal_pos_w[0, 0], torch.tensor([0., 0.02, 0.05]))
        torch.testing.assert_close(cloth.data.nodal_vel_w[0, 0], torch.tensor([0., 2., 5.]))
        torch.testing.assert_close(cloth.data.nodal_pos_w[0, 1], torch.tensor([0.1, 0., 0.]))
        torch.testing.assert_close(cloth.data.nodal_pos_w[1, 0], torch.tensor([1., 0., 0.]))

    def test_release_stops_surface_writes(self):
        cloth = Cloth()
        anchor = torch.zeros(3)
        picker = VirtualPicker(cloth, lambda: anchor)
        picker.grasp(anchor)
        picker.release()
        writes = cloth.writes
        anchor[2] = 1.
        picker.update(dt=0.01)
        self.assertEqual(cloth.writes, writes)
        self.assertFalse(picker.is_grasping)

    def test_rejects_distant_gripper_and_distant_selection(self):
        cloth = Cloth()
        picker = VirtualPicker(cloth, lambda: torch.ones(3))
        self.assertFalse(picker.grasp([0., 0., 0.]))
        picker = VirtualPicker(cloth, lambda: torch.zeros(3))
        self.assertFalse(picker.grasp([1., 1., 1.]))
        self.assertFalse(picker.is_grasping)

    def test_kinematic_reset_releases_only_owned_vertex(self):
        cloth = Cloth(kinematic=True)
        cloth.data.nodal_kinematic_target[0, 1, 3] = 0.
        anchor = torch.zeros(3)
        picker = VirtualPicker(cloth, lambda: anchor)
        picker.grasp(anchor)
        anchor[2] = 0.1
        picker.update()
        self.assertEqual(float(cloth.data.nodal_kinematic_target[0, 0, 3]), 0.)
        self.assertAlmostEqual(float(cloth.data.nodal_kinematic_target[0, 0, 2]), 0.1)
        picker.reset()
        self.assertEqual(float(cloth.data.nodal_kinematic_target[0, 0, 3]), 1.)
        self.assertEqual(float(cloth.data.nodal_kinematic_target[0, 1, 3]), 0.)
        anchor.zero_()  # model the robot reset before the next episode
        self.assertTrue(picker.grasp([0., 0., 0.]))  # new episode / re-grasp

    def test_validation_and_existing_constraint(self):
        cloth = Cloth(kinematic=True)
        with self.assertRaises(ValueError):
            create_picker(cloth, lambda: torch.zeros(3), grasp_mode="physical")
        picker = VirtualPicker(cloth, lambda: torch.zeros(3))
        with self.assertRaises(ValueError):
            picker.grasp([0., 0., 0.], vertex_id=-1)
        cloth.data.nodal_kinematic_target[0, 0, 3] = 0.
        with self.assertRaises(RuntimeError):
            picker.grasp([0., 0., 0.])

    def test_selected_environment_only(self):
        cloth = Cloth()
        anchor = torch.tensor([1., 0., 0.])
        picker = VirtualPicker(cloth, lambda: anchor, env_id=1)
        picker.grasp(anchor)
        anchor[2] = 0.1
        picker.update(dt=0.01)
        self.assertAlmostEqual(float(cloth.data.nodal_pos_w[1, 0, 2]), 0.1)
        self.assertEqual(float(cloth.data.nodal_pos_w[0, 0, 2]), 0.)


if __name__ == "__main__":
    unittest.main()
