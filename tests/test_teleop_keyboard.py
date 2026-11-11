"""Keyboard state checks without creating Kit or a physics environment."""
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/grasp"))
from teleop_cloth import KeyboardInput, ClothTeleop
from physical_picker import PhysicalPicker


class TestKeyboardInput(unittest.TestCase):
    def setUp(self):
        self.keyboard = KeyboardInput.__new__(KeyboardInput)
        self.keyboard.carb = SimpleNamespace(
            KeyboardEventType=SimpleNamespace(KEY_PRESS=0, KEY_RELEASE=2),
        )
        self.keyboard.held = set()
        self.keyboard.events = deque()

    def event(self, key, kind=0, *, text=False):
        value = key if text else SimpleNamespace(name=key)
        self.keyboard.on_event(SimpleNamespace(input=value, type=kind))

    def test_hold_repeat_and_unmatched_release(self):
        self.event("W")
        self.event("W")
        self.event("W", 1)  # OS repeat
        self.assertEqual(self.keyboard.direction(), [1, 0, 0])
        self.event("S")
        self.assertEqual(self.keyboard.direction(), [0, 0, 0])
        self.event("W", 2)
        self.event("S", 2)
        self.event("W", 2)
        self.assertEqual(self.keyboard.direction(), [0, 0, 0])

    def test_one_toggle_per_press_and_reset_release(self):
        self.event("K")
        self.event("K")
        self.event("K", 1)
        self.event("K", 2)
        self.event("K")
        self.assertEqual(list(self.keyboard.events), ["K", "K"])
        self.event("Q")
        self.keyboard.held.clear()
        self.event("Q", 2)
        self.assertEqual(self.keyboard.direction(), [0, 0, 0])

    def test_string_key_press_release_and_toggle(self):
        self.event("w", text=True)
        self.assertEqual(self.keyboard.direction(), [1, 0, 0])
        self.event("W", 2, text=True)
        self.assertEqual(self.keyboard.direction(), [0, 0, 0])
        self.event("k", text=True)
        self.event("k", 1, text=True)
        self.assertEqual(list(self.keyboard.events), ["K"])
        self.event("k", 2, text=True)
        self.event("k", text=True)
        self.assertEqual(list(self.keyboard.events), ["K", "K"])

    def test_character_text_is_ignored(self):
        # Kit sends these in addition to physical key events. They must neither
        # throw AttributeError nor latch movement or queue another toggle.
        self.event("w", 3, text=True)
        self.event("k", 3, text=True)
        self.assertEqual(self.keyboard.direction(), [0, 0, 0])
        self.assertEqual(list(self.keyboard.events), [])


class TestManualPhysicalControl(unittest.TestCase):
    def setUp(self):
        self.session = ClothTeleop.__new__(ClothTeleop)
        s = self.session
        s.args = SimpleNamespace(grasp_mode="physical", speed=.06, max_episode_seconds=600)
        s.cloth = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=torch.zeros((1, 4, 3))))
        s.target = torch.tensor([.5, 0., .1])
        s.tcp = lambda: s.target.clone()
        s.corners = lambda: s.cloth.data.nodal_pos_w[0].clone()
        s.picker = PhysicalPicker(s.cloth, s.tcp, grasp_radius=.025)
        s.action = torch.zeros((1, 8))
        s.action[0, 7] = 1
        s.recorder = Mock()
        s.dt, s.steps = .01, 0
        s.mode, s.finished = "ready", False
        s.grasp_verified = False
        s.grasp_checks = []
        s.completed_actions = 0
        def step(**unused):
            s.steps += 1
            s.observe_physical_grasp()
        s.step = step

    def test_far_close_and_failed_estimate_do_not_take_control(self):
        s = self.session
        before = s.target.clone()
        s.toggle_grasp()  # No cloth nearby: fingers must still close.
        self.assertEqual(float(s.action[0, 7]), -1)
        self.assertFalse(s.picker.is_grasping)
        for _ in range(200):
            s.tick()
        self.assertTrue(torch.equal(s.target, before))
        self.assertFalse(s.grasp_verified)
        self.assertEqual(float(s.action[0, 7]), -1)
        s.tick([0, 0, 1])
        self.assertGreater(float(s.target[2]), float(before[2]))
        s.toggle_grasp()
        self.assertEqual(float(s.action[0, 7]), 1)
        before_open_move = s.target.clone()
        s.tick([1, 0, 0])
        self.assertGreater(float(s.target[0]), float(before_open_move[0]))
        self.assertEqual(s.mode, "ready")
        self.assertEqual(s.completed_actions, 1)
        s.recorder.end_pick_place.assert_called_once()

    def test_passive_estimate_loss_does_not_open_fingers_or_move_arm(self):
        s = self.session
        s.cloth.data.nodal_pos_w[0, :, :] = s.target
        s.toggle_grasp()
        s.target[2] += .03
        s.cloth.data.nodal_pos_w[0, :, 2] += .03
        s.observe_physical_grasp()
        self.assertTrue(s.grasp_verified)
        s.target[2] += .05  # Hand moves away while material stays behind.
        before = s.target.clone()
        s.observe_physical_grasp()
        self.assertFalse(s.grasp_verified)
        self.assertTrue(torch.equal(s.target, before))
        self.assertEqual(float(s.action[0, 7]), -1)
        self.assertEqual(s.mode, "carry")


if __name__ == "__main__":
    unittest.main()
