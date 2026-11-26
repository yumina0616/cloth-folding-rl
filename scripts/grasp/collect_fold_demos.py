"""Run the physical fold demo repeatedly until enough successful demonstrations are collected.

Each run is its own Isaac process with the next seed, so --pos_jitter / --yaw_jitter give every run
a different cloth position and rotation. Every run folder (successful or not) goes into one
collection folder; successful runs hold the BC dataset files (high_level.jsonl, low_level.npz,
metadata.json, context.json) that scripts/learning/demonstration_dataset.py reads.

    data/demonstrations/physical_fold/<collection time>/
        collect_settings.json   arguments, git commit
        collect_summary.csv     one row per run (written right after each run)
        logs/seed_0000.log      full output of each run
        <run time>_ok/ ...      run folders written by rectangular_fold_demo.py

A stopped collection resumes when started again with --resume <collection folder>.

This script does not import Isaac Lab. Run it with the Isaac Lab Python environment active
(the launcher it calls uses that environment), from the repository root, for example:

    python3 scripts/grasp/collect_fold_demos.py --successes 20 --pos_jitter 0.02 --yaw_jitter 15 \\
        --success_distance 0.05 --shape_iou_threshold 0 \\
        --table_friction 0.6 --table_static_friction 0.65 presets=isaacsim_physx

Options this script does not know are passed to rectangular_fold_demo.py unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "scripts" / "grasp" / "rectangular_fold_demo.py"
RESULT_PREFIX = "[fold] RESULT "
COLUMNS = ["seed", "status", "success", "final_mm_1", "final_mm_2", "iou", "cloth_x", "cloth_y",
           "yaw_cmd_deg", "yaw_measured_deg", "attempts", "corrections", "seconds", "run_dir", "log"]
# Set by this script for every run; passing them again would silently change what a run means
RESERVED = ("--seed", "--video_dir", "--pos_jitter", "--yaw_jitter", "--stop_after")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--successes", type=int, required=True, help="Stop after this many successful runs")
    parser.add_argument("--max_runs", type=int, default=None, help="Stop after this many runs in total (default 3 x successes)")
    parser.add_argument("--pos_jitter", type=float, required=True, help="Cloth center shift range, x and y each +- [m]")
    parser.add_argument("--yaw_jitter", type=float, required=True, help="Cloth rotation range +- [deg] (< 45)")
    parser.add_argument("--start_seed", type=int, default=0, help="Seed of the first run; each run adds 1")
    parser.add_argument("--grasp_mode", choices=["physical", "virtual"], default="physical")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Collection folder (default data/demonstrations/<grasp_mode>_fold/<now>)")
    parser.add_argument("--resume", type=Path, default=None, help="Continue a stopped collection folder")
    parser.add_argument("--isaaclab", default=os.environ.get("ISAACLAB_LAUNCHER", "~/IsaacLab/isaaclab.sh"),
                        help="Isaac Lab launcher (isaaclab.sh / isaaclab.bat); env ISAACLAB_LAUNCHER")
    parser.add_argument("--run_timeout_s", type=float, default=1800, help="Kill a run that takes longer [s]")
    parser.add_argument("--max_crashes_in_a_row", type=int, default=2,
                        help="Stop when this many runs in a row end without a result (setup problem, not a fold)")
    args, demo_args = parser.parse_known_args(argv)
    demo_args = [a for a in demo_args if a != "--"]
    for name in RESERVED:
        if any(a == name or a.startswith(name + "=") for a in demo_args):
            parser.error(f"{name} is set by this script")
    if args.successes < 1:
        parser.error("--successes must be at least 1")
    if not 0 <= args.yaw_jitter < 45 or args.pos_jitter < 0:
        parser.error("--pos_jitter must be >= 0 and --yaw_jitter in [0, 45)")
    if args.max_runs is None:
        args.max_runs = 3 * args.successes
    return args, demo_args


def launcher_command(launcher):
    path = str(Path(launcher).expanduser())
    # A .py launcher (used by the tests) runs with this Python
    return [sys.executable, path] if path.endswith(".py") else [path]


def parse_result(log_text):
    """The run summary printed by rectangular_fold_demo.py, or None if the run crashed before it."""
    for line in reversed(log_text.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return None


def summary_row(seed, summary, seconds, log_path):
    row = dict.fromkeys(COLUMNS, "")
    row.update(seed=seed, seconds=round(seconds, 1), log=str(log_path))
    if summary is None:
        row.update(status="crashed", success=False)
        return row
    detail = {}
    run_dir = summary.get("run_dir")
    if run_dir and (Path(run_dir) / "result.json").is_file():
        detail = json.loads((Path(run_dir) / "result.json").read_text(encoding="utf-8"))
    distances = summary.get("final_distances_m") or []
    start = summary.get("cloth_start") or ["", ""]
    grasps = summary.get("grasps") or []
    row.update(
        status=summary.get("status"), success=bool(summary.get("success")),
        final_mm_1=round(distances[0] * 1000, 1) if len(distances) > 0 else "",
        final_mm_2=round(distances[1] * 1000, 1) if len(distances) > 1 else "",
        iou=round((detail.get("shape_metrics") or {}).get("iou"), 4) if (detail.get("shape_metrics") or {}).get("iou") is not None else "",
        cloth_x=start[0], cloth_y=start[1],
        yaw_cmd_deg=detail.get("cloth_start_yaw_deg", ""), yaw_measured_deg=detail.get("cloth_yaw_measured_deg", ""),
        attempts=" ".join(str(g.get("attempts")) for g in grasps),
        corrections=detail.get("correction_count", ""), run_dir=run_dir or "")
    return row


def read_done(csv_path):
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    args, demo_args = parse_args(argv)
    if args.resume is not None:
        out_dir = args.resume.resolve()
        if not (out_dir / "collect_summary.csv").is_file():
            sys.exit(f"Nothing to resume in {out_dir}")
    else:
        out_dir = (args.out_dir or ROOT / "data" / "demonstrations" / f"{args.grasp_mode}_fold"
                   / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "collect_settings.json").write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "demo_args": demo_args,
             "git_commit": git_commit(), "started": datetime.now().isoformat()}, indent=2), encoding="utf-8")
    (out_dir / "logs").mkdir(exist_ok=True)
    csv_path = out_dir / "collect_summary.csv"
    done = read_done(csv_path)
    successes = sum(r["success"] == "True" for r in done)
    seed = max((int(r["seed"]) for r in done), default=args.start_seed - 1) + 1
    print(f"[collect] folder {out_dir}: {len(done)} runs, {successes} successes so far", flush=True)
    if not os.environ.get("VIRTUAL_ENV") and not os.environ.get("CONDA_PREFIX"):
        print("[collect] WARNING: no Python environment is active; activate the Isaac Lab environment first "
              "(otherwise the launcher may fail with ModuleNotFoundError: tomllib)", flush=True)
    crashes_in_a_row = 0

    while successes < args.successes and len(done) < args.max_runs:
        log_path = out_dir / "logs" / f"seed_{seed:04d}.log"
        command = [*launcher_command(args.isaaclab), "-p", str(DEMO), "--grasp_mode", args.grasp_mode,
                   "--save_all_videos", "--video_dir", str(out_dir), "--seed", str(seed),
                   "--pos_jitter", str(args.pos_jitter), "--yaw_jitter", str(args.yaw_jitter), *demo_args]
        print(f"[collect] run {len(done) + 1}/{args.max_runs} seed {seed} "
              f"(successes {successes}/{args.successes})", flush=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            try:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.run_timeout_s)
            except subprocess.TimeoutExpired:
                log.write(f"\n[collect] killed after {args.run_timeout_s} s\n")
        summary = parse_result(log_path.read_text(encoding="utf-8", errors="replace"))
        row = summary_row(seed, summary, time.monotonic() - started, log_path)
        new_file = not csv_path.is_file()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        done.append({k: str(v) for k, v in row.items()})
        successes += bool(row["success"])
        crashes_in_a_row = crashes_in_a_row + 1 if row["status"] == "crashed" else 0
        if row["status"] == "crashed":
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            print("[collect]   last lines of " + str(log_path) + ":
    " + "
    ".join(tail), flush=True)
        print(f"[collect]   -> {row['status']} success={row['success']} final_mm={row['final_mm_1']},"
              f"{row['final_mm_2']} IoU={row['iou']} yaw={row['yaw_cmd_deg']}", flush=True)
        seed += 1
        if crashes_in_a_row >= args.max_crashes_in_a_row:
            print(f"[collect] stopping: {crashes_in_a_row} runs in a row ended without a result; "
                  "fix the error above and continue with --resume", flush=True)
            break

    ious = [float(r["iou"]) for r in done if r["success"] == "True" and r["iou"]]
    print(f"[collect] done: {successes} successes in {len(done)} runs "
          f"({100 * successes / max(len(done), 1):.0f}%)"
          + (f", IoU mean of successes {sum(ious) / len(ious):.3f}" if ious else "")
          + f"\n[collect] summary: {csv_path}", flush=True)
    return 0 if successes >= args.successes else 1


if __name__ == "__main__":
    sys.exit(main())
