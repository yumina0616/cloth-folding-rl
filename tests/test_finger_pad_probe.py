import importlib.util
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "finger_pad_probe", Path(__file__).resolve().parents[1] / "scripts" / "grasp" / "finger_pad_probe.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
tip_extents = MODULE.tip_extents


class TipExtentsTest(unittest.TestCase):
    def test_tip_slice_is_only_the_lowest_part(self):
        # A finger 18 x 22 mm wide above z = 0.011, with a narrower 8 x 12 mm tip below z = 0.01
        body = [(x, y, z) for x in (0.0, 0.018) for y in (0.0, 0.022) for z in (0.011, 0.054)]
        tip = [(x, y, z) for x in (0.005, 0.013) for y in (0.005, 0.017) for z in (0.0, 0.01)]
        ext = tip_extents(body + tip, tip_height=0.01)
        self.assertTrue(np.allclose(ext["whole_size"], (0.018, 0.022, 0.054)))
        self.assertTrue(np.allclose(ext["tip_size"], (0.008, 0.012, 0.01)))

    def test_rejects_empty_points(self):
        with self.assertRaises(ValueError):
            tip_extents(np.empty((0, 3)))


if __name__ == "__main__":
    unittest.main()
