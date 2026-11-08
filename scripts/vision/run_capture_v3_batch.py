"""Launch repeated Isaac Lab v3.0 cloth demos until 300 synchronized camera pairs.

Run from cloth-folding-rl repo root, e.g.:
  python scripts/vision/run_capture_v3_batch.py --launcher './isaaclab.sh -p'
"""
import argparse
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True, help="Your existing Isaac Lab launcher prefix")
    parser.add_argument("--target", type=int, default=300, help="Target synchronized frame pairs")
    parser.add_argument("--per_run", type=int, default=15)
    parser.add_argument("--max_runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--capture_dir", default="data/vision/raw")
    parser.add_argument("--wrist_parent", default=None, help="Only if default nested panda_hand USD path differs")
    args = parser.parse_args()
    script = Path("scripts/grasp/rectangular_fold_capture_v3.py")
    if not script.exists():
        parser.error("Run from cloth-folding-rl repository root after copying script to scripts/grasp")
    if min(args.target, args.per_run, args.max_runs) <= 0:
        parser.error("target/per_run/max_runs must be positive")
    root = Path(args.capture_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = root / f"capture_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    captured_total = 0
    for run in range(args.max_runs):
        if captured_total >= args.target:
            break
        remaining = args.target - captured_total
        per_run = min(args.per_run, remaining)
        x = round(rng.uniform(0.44, 0.56), 4)
        y = round(rng.uniform(-0.12, 0.03), 4)
        yaw = round(rng.uniform(-30, 30), 2)
        # Each run contains a new, independently positioned cloth and a full fold.
        cmd = shlex.split(args.launcher, posix=(sys.platform != "win32")) + [
            str(script), "--no_video", "--cloth_x", str(x), "--cloth_y", str(y),
            "--cloth_yaw_deg", str(yaw), "--seed", str(args.seed + run),
            "--capture_limit", str(per_run), "--capture_stride", "75",
            "--capture_dir", str(root),
        ]
        if args.wrist_parent:
            cmd.extend(["--wrist_parent", args.wrist_parent])
        existing = {str(p) for p in root.glob("*/capture_summary.json")}
        print(f"[{run+1}/{args.max_runs}] {captured_total}/{args.target} pairs; x={x}, y={y}, yaw={yaw}", flush=True)
        try:
            completed = subprocess.run(cmd, check=False)
            code = completed.returncode
        except OSError as exc:
            code = -1
            print(f"Could not launch Isaac Lab: {exc}", file=sys.stderr)
        new_summaries = [p for p in root.glob("*/capture_summary.json") if str(p) not in existing]
        new_pairs = 0
        for summary in new_summaries:
            try:
                new_pairs += int(json.loads(summary.read_text(encoding="utf-8")).get("saved_pairs", 0))
            except (OSError, ValueError, TypeError):
                pass
        captured_total += new_pairs
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run": run+1, "exit_code": code, "saved_pairs": new_pairs,
                                "total": captured_total, "cloth_x": x, "cloth_y": y, "cloth_yaw": yaw}) + "\n")
        if code != 0:
            print("[WARNING] Run failed. Fix logged error before continuing more GPU runs.", flush=True)
            break
        if new_pairs == 0:
            print("[WARNING] No samples captured. Stopping to avoid unnecessary GPU costs.", flush=True)
            break
    print(f"Finished: {captured_total}/{args.target} paired timepoints ({2*captured_total} RGB images).")
    print(f"Log: {manifest}")
    print("Important: Pixel keypoints and occlusion are UNVERIFIED; do not train YOLO before overlay review.")


if __name__ == "__main__":
    main()
