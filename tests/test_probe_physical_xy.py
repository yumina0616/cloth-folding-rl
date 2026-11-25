"""XY probe planning, result validation, and bounded/resumable orchestration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.grasp import probe_physical_xy as probe


def expert_result(trial, success=True):
    return {"grasp_mode": "physical", "stop_after": "fold", "cloth_start": [trial["x"], trial["y"]],
            "status": "completed" if success else "grasp_failed", "success": success,
            "final_distances_m": [.01, .015] if success else None, "shape_metrics": {"iou": .9}}


class ProbeTest(unittest.TestCase):
    def test_default_plan_is_zero_yaw_and_baseline_first(self):
        plan = probe.make_plan()
        self.assertEqual(len(plan["trials"]), 27)
        self.assertTrue(all(t["baseline"] for t in plan["trials"][:3]))
        self.assertEqual(plan["yaw_deg"], 0)
        self.assertEqual(plan["pos_jitter"], 0)
        self.assertEqual({t["x"] for t in plan["trials"]}, {.47, .5, .53})
        self.assertEqual({t["y"] for t in plan["trials"]}, {-.08, -.05, -.02})

    def test_invalid_positions_and_overrides_rejected(self):
        for kwargs in ({"offsets": [float("nan"), 0]}, {"offsets": [.01]}, {"offsets": [-.3, 0, .3]},
                       {"repeats": 0}, {"params": {"pos_jitter": .1}},
                       {"params": {"cloth_yaw_deg": 20}}, {"params": {"cloth_density": float("inf")}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                probe.make_plan(**kwargs)

    def test_command_is_physical_full_fold_without_extra_noise(self):
        plan = probe.make_plan(params={"finger_friction": 2.5})
        command = probe.command_for(plan, plan["trials"][0], Path("output"))
        self.assertEqual(command[command.index("--grasp_mode") + 1], "physical")
        self.assertEqual(command[command.index("--stop_after") + 1], "fold")
        self.assertEqual(command[command.index("--cloth_yaw_deg") + 1], "0")
        self.assertEqual(command[command.index("--pos_jitter") + 1], "0")
        self.assertIn("--no_video", command)

    def test_video_includes_failed_attempts(self):
        plan = probe.make_plan(video=True)
        command = probe.command_for(plan, plan["trials"][0], Path("output"))
        self.assertIn("--save_all_videos", command)
        self.assertNotIn("--no_video", command)

    def test_old_nonvideo_plan_ignores_only_orchestrator_fingerprint(self):
        plan = probe.make_plan()
        old = json.loads(json.dumps(plan))
        old.pop("video")
        old["sources"][str(Path(probe.__file__).resolve())] = "old orchestration code"
        self.assertEqual(probe.comparable_plan(plan), probe.comparable_plan(old))
        old["sources"][old["expert"]] = "changed expert"
        self.assertNotEqual(probe.comparable_plan(plan), probe.comparable_plan(old))

    def test_reference_profile_and_disabled_iou_reproduction(self):
        plan = probe.make_plan()
        self.assertEqual(plan["params"]["pinch_jaw_shift"], .01)
        self.assertEqual(plan["params"]["check_lift"], .02)
        self.assertEqual(plan["params"]["shape_iou_threshold"], .8)
        command = probe.command_for(plan, plan["trials"][0], Path("output"))
        self.assertIn("--retry_in_place", command)
        self.assertNotIn("True", command)
        self.assertEqual(probe.make_plan(iou_min=0)["params"]["shape_iou_threshold"], 0)

    def test_iou_zero_records_low_iou_without_rejecting_corner_success(self):
        trial = probe.make_plan()["trials"][0]
        result = expert_result(trial)
        result["shape_metrics"]["iou"] = .2
        self.assertTrue(probe.parse_result("[fold] RESULT " + json.dumps(result), trial, iou_min=0)["success"])

    def test_result_must_match_request_and_both_distances(self):
        trial = probe.make_plan()["trials"][0]
        result = expert_result(trial)
        log = "[fold] RESULT " + json.dumps(result)
        self.assertTrue(probe.parse_result(log, trial)["success"])
        for changed in ({"cloth_start": [.6, 0]}, {"stop_after": "transport"},
                        {"final_distances_m": [.01, .06]}, {"final_distances_m": [float("nan"), .01]},
                        {"shape_metrics": {"iou": .79}}, {"shape_metrics": None}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                probe.parse_result("[fold] RESULT " + json.dumps({**result, **changed}), trial)
        with self.assertRaises(ValueError):
            probe.parse_result("Simulation failed before creating an environment", trial)

    def test_partial_points_are_not_marked_as_passing(self):
        plan = probe.make_plan()
        trial = plan["trials"][0]
        result = probe.summarize(plan, {trial["id"]: {**trial, "success": True, "process_status": "OK"}})
        self.assertFalse(result["points"][0]["candidate"])
        self.assertIsNone(result["points"][1]["success_rate"])

    def main(self, output, extra=(), trial_runner=None):
        argv = ["probe_physical_xy", "--out", str(output), "--repeats", "1", *extra]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            if trial_runner is None:
                probe.main()
            else:
                with patch.object(probe, "run_trial", side_effect=trial_runner):
                    probe.main()

    def test_plan_only_never_starts_a_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "run_trial") as runner:
            self.main(Path(directory))
            runner.assert_not_called()
            self.assertTrue((Path(directory) / "plan.json").exists())
            self.assertFalse((Path(directory) / "results.jsonl").exists())

    def test_resume_skips_completed_trials(self):
        calls = []
        def run(plan, trial, output, timeout):
            calls.append(trial["id"])
            return {**trial, "success": True, "process_status": "OK"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.main(output, ["--run", "--max_trials", "1"], run)
            self.main(output, ["--run", "--max_trials", "1"], run)
            self.assertEqual(calls, ["p000_r00", "p001_r00"])
            self.assertFalse((output / ".running.lock").exists())
            with self.assertRaises(SystemExit):
                self.main(output, ["--offsets", "-.05", "0", ".05"])

    def test_failed_baseline_stops_other_positions(self):
        calls = []
        def run(plan, trial, output, timeout):
            calls.append(trial["id"])
            return {**trial, "success": False, "process_status": "OK"}
        with tempfile.TemporaryDirectory() as directory:
            self.main(Path(directory), ["--run"], run)
            self.assertEqual(calls, ["p000_r00"])


if __name__ == "__main__":
    unittest.main()
