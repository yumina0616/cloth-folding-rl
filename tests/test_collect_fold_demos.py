"""collect_fold_demos.py with a fake Isaac launcher; runs without Isaac Sim."""
import csv
import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "collect_fold_demos_under_test", Path(__file__).parents[1] / "scripts" / "grasp" / "collect_fold_demos.py")
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)

# Fake launcher: seed % 3 == 0 -> success, 1 -> fail, 2 -> crash without a RESULT line
FAKE = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    argv = sys.argv
    seed = int(argv[argv.index("--seed") + 1]); out = Path(argv[argv.index("--video_dir") + 1])
    if seed % 3 == 2:
        sys.exit("boom")
    ok = seed % 3 == 0
    run = out / f"run_{seed}_{'ok' if ok else 'fail'}"; run.mkdir()
    (run / "result.json").write_text(json.dumps({"shape_metrics": {"iou": 0.85}, "cloth_start_yaw_deg": 3.0,
                                                 "cloth_yaw_measured_deg": 3.1, "correction_count": 0}))
    print("[fold] RESULT " + json.dumps({"status": "completed", "success": ok, "final_distances_m": [0.02, 0.01],
          "cloth_start": [0.5, -0.05], "grasps": [{"attempts": 1}, {"attempts": 2}], "run_dir": str(run)}))
''')


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fake = self.dir / "fake_isaaclab.py"
        self.fake.write_text(FAKE)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, out):
        with (out / "collect_summary.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_stops_at_target_successes_and_records_every_run(self):
        out = self.dir / "collect"
        code = COLLECT.main(["--successes", "2", "--pos_jitter", "0.02", "--yaw_jitter", "15",
                             "--isaaclab", str(self.fake), "--out_dir", str(out), "--success_distance", "0.05"])
        self.assertEqual(code, 0)
        rows = self.rows(out)
        self.assertEqual([r["seed"] for r in rows], ["0", "1", "2", "3"])  # successes at seeds 0 and 3
        self.assertEqual([r["status"] for r in rows], ["completed", "completed", "crashed", "completed"])
        self.assertEqual(rows[0]["iou"], "0.85")
        self.assertEqual(rows[0]["attempts"], "1 2")
        self.assertTrue((out / "logs" / "seed_0002.log").is_file())
        settings = json.loads((out / "collect_settings.json").read_text())
        self.assertEqual(settings["demo_args"], ["--success_distance", "0.05"])

    def test_resume_continues_with_the_next_seed(self):
        out = self.dir / "collect"
        COLLECT.main(["--successes", "1", "--pos_jitter", "0", "--yaw_jitter", "0",
                      "--isaaclab", str(self.fake), "--out_dir", str(out)])
        COLLECT.main(["--successes", "2", "--pos_jitter", "0", "--yaw_jitter", "0",
                      "--isaaclab", str(self.fake), "--resume", str(out)])
        self.assertEqual([r["seed"] for r in self.rows(out)], ["0", "1", "2", "3"])

    def test_max_runs_and_reserved_options(self):
        out = self.dir / "collect"
        code = COLLECT.main(["--successes", "5", "--max_runs", "2", "--pos_jitter", "0", "--yaw_jitter", "0",
                             "--isaaclab", str(self.fake), "--out_dir", str(out)])
        self.assertEqual(code, 1)
        self.assertEqual(len(self.rows(out)), 2)
        with self.assertRaises(SystemExit):
            COLLECT.parse_args(["--successes", "1", "--pos_jitter", "0", "--yaw_jitter", "0", "--seed", "3"])


if __name__ == "__main__":
    unittest.main()
