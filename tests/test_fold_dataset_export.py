"""Fold demo -> PR #37 BC dataset format; runs without Isaac Sim."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))  # scripts.learning.teleop_recorder
SPEC = importlib.util.spec_from_file_location("fold_dataset_export_under_test",
                                              ROOT / "scripts" / "grasp" / "fold_dataset_export.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PAIRS = [(0, 1), (2, 3)]
FLAT = np.array([[.4, -.2, 0], [.4, .1, 0], [.6, -.2, 0], [.6, .1, 0]], dtype=float)
FOLDED = np.array([[.4, .1, .003], [.4, .1, 0], [.6, .1, .003], [.6, .1, 0]], dtype=float)


class FoldDatasetExportTest(unittest.TestCase):
    def run_steps(self, log, count, detail, grasping=False):
        for _ in range(count):
            self.t += 0.01
            log.append(self.t, detail, np.r_[.5, 0, .1, 1, 0, 0, 0, 1], np.zeros(9), FLAT, [.5, 0, .1], grasping)

    def make_log(self, with_correction=False):
        self.t = 0.0
        log = MODULE.FoldEpisodeLog()
        self.run_steps(log, 3, "1_approach")
        # Failed attempt: close, lift check fails, open
        log.begin(FLAT, [.41, -.17])
        self.run_steps(log, 2, "1_grasp")
        log.start_carry()
        self.run_steps(log, 2, "1_grasp_check", grasping=True)
        log.abandon()
        self.run_steps(log, 2, "1_grasp")
        # Successful attempts for both pairs (and optionally a correction of pair 1)
        picks = [(0, "1"), (1, "2")] + ([(1, "correction_1")] if with_correction else [])
        for pair_index, number in picks:
            log.begin(FLAT, [.41, -.17])
            self.run_steps(log, 2, f"{number}_grasp")
            log.start_carry()
            self.run_steps(log, 4, f"{number}_fold_arc_1", grasping=True)
            log.begin_release([.41, .07])
            self.run_steps(log, 2, f"{number}_release")
            log.end(FOLDED, pair_index, correction=number.startswith("correction"))
            self.run_steps(log, 2, f"{number}_retreat")
        return log

    def save(self, log, folder):
        context = MODULE.fold_context(grasp_mode="physical", corner_vertex_indices=[1, 2, 3, 4], pairs=PAIRS,
                                      sample_dt=.01, success_threshold_m=.05,
                                      final_pair_distances_m=[.003, .003], config={})
        return log.save(folder, PAIRS, .05, FOLDED, "episode", context)

    def test_writes_the_four_loader_files_with_loader_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(self.save(self.make_log(), tmp))
            folder = Path(tmp)
            for name in ("high_level.jsonl", "low_level.npz", "metadata.json", "context.json"):
                self.assertTrue((folder / name).is_file(), name)
            records = [json.loads(line) for line in (folder / "high_level.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)  # the failed attempt is not a pick/place
            self.assertEqual([r["pair_index"] for r in records], [0, 1])
            with np.load(folder / "low_level.npz") as low:
                phases = [str(p) for p in low["phase"]]
                self.assertEqual(low["action"].shape[1], 8)
                for record in records:
                    segment = phases[record["low_level_start"]:record["low_level_end"]]
                    self.assertEqual(segment[0], "close")
                    self.assertIn("carry", segment)
                    self.assertEqual(segment[-1], "physical_release")
            self.assertEqual(phases.count("grasp_retry"), 4)  # close 2 + failed check 2
            context = json.loads((folder / "context.json").read_text())
            self.assertEqual(context["pick_xy"], "measured TCP at manual close")
            self.assertEqual(context["place_xy"], "measured TCP immediately before opening")
            detail = np.load(folder / "low_level_phase_detail.npy")
            self.assertEqual(len(detail), len(phases))

    def test_correction_is_an_extra_marked_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(self.save(self.make_log(with_correction=True), tmp))
            records = [json.loads(line) for line in (Path(tmp) / "high_level.jsonl").read_text().splitlines()]
            self.assertEqual([r["correction"] for r in records], [False, False, True])
            self.assertEqual(records[-1]["pair_index"], 1)

    def test_failed_fold_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self.make_log()
            context = MODULE.fold_context(grasp_mode="physical", corner_vertex_indices=[1, 2, 3, 4], pairs=PAIRS,
                                          sample_dt=.01, success_threshold_m=.05,
                                          final_pair_distances_m=[.3, .3], config={})
            self.assertFalse(log.save(tmp, PAIRS, .05, FLAT, "episode", context))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_open_attempt_is_rejected(self):
        log = MODULE.FoldEpisodeLog()
        log.begin(FLAT, [.4, -.2])
        with self.assertRaises(RuntimeError):
            log.begin(FLAT, [.4, -.2])


if __name__ == "__main__":
    unittest.main()
