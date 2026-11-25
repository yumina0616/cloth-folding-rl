"""Probe full physical folds on an XY grid with yaw=0 and fixed physics.

Default: write/print a plan only. Add --run to launch trials sequentially.
Launch with Isaac's Python, e.g. python.bat scripts/grasp/probe_physical_xy.py.
Output: plan.json, results.jsonl, summary.json, summary.csv, trial logs/artifacts.
This probes the expert's range; demonstration.npz is NOT the aligned BC format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXPERT = Path(__file__).with_name("rectangular_fold_demo.py")
PROFILE = {"success_distance": .05, "shape_iou_threshold": .8, "pinch_jaw_shift": .01,
    "close_lift": .002, "check_lift": .02, "arc_peak": .04, "table_friction": .6,
    "table_static_friction": .65, "release_height": .04, "place_inset": .01,
    "move_axis": "y", "free_speed": 2., "retry_in_place": True, "place_settle_s": .5}
ALLOWED = {"lift_height", "settle_seconds", "success_distance", "cloth_size", "mesh_refinement",
    "cloth_thickness", "cloth_density", "youngs_modulus", "bend_stiffness", "poissons_ratio",
    "damping", "cloth_friction", "table_friction", "table_length", "table_width", "open_width",
    "pad_half", "edge_margin", "press_depth", "finger_friction", "gripper_effort", "check_lift",
    "check_hold_s", "check_min_rise", "quality_max", "place_mode", "grasp_yaw_deg", "slip_drift",
    "drop_height", "drop_hand_height"}
ALLOWED |= set(PROFILE)


def make_plan(*, center_x=.5, center_y=-.05, offsets=(-.03, 0., .03), repeats=3,
              seed=0, retries=1, params=None, expert=EXPERT, video=False, iou_min=.8):
    params = {**PROFILE, **dict(params or {}), "shape_iou_threshold": iou_min}
    if set(params) - ALLOWED:
        raise ValueError(f"Unsupported/protected parameters: {sorted(set(params) - ALLOWED)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for k, v in params.items() if k not in ("place_mode", "move_axis", "retry_in_place")):
        raise ValueError("Expert numeric parameters must be finite numbers")
    if not math.isfinite(iou_min) or not 0 <= iou_min <= 1:
        raise ValueError("iou_min must be in [0,1]")
    if params["move_axis"] not in ("auto", "x", "y") or not isinstance(params["retry_in_place"], bool):
        raise ValueError("Invalid move_axis or retry_in_place")
    if "place_mode" in params and params["place_mode"] not in ("mirror", "live"):
        raise ValueError("place_mode must be mirror or live")
    offsets = sorted(set(offsets))
    if (not offsets or not all(math.isfinite(v) for v in (*offsets, center_x, center_y))
            or 0. not in offsets):
        raise ValueError("Finite offsets must include zero (the baseline)")
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(retries, int) or retries < 0:
        raise ValueError("repeats must be positive and retries nonnegative")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    size = params.get("cloth_size", .3)
    if size <= 0:
        raise ValueError("cloth_size must be positive")
    positions = sorted(((dx, dy) for dx in offsets for dy in offsets),
                       key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
    trials = []
    for point, (dx, dy) in enumerate(positions):
        x, y = center_x + dx, center_y + dy
        # Conservative XY envelope shared with the policy. This is not an IK proof.
        if x - size / 2 < .15 - 1e-6 or x + size / 2 > .78 + 1e-6 or y - size / 2 < -.42 - 1e-6 or y + size / 2 > .42 + 1e-6:
            raise ValueError(f"Cloth at ({x:.3f}, {y:.3f}) leaves the policy XY envelope")
        for repeat in range(repeats):
            trials.append({"id": f"p{point:03d}_r{repeat:02d}", "point": point, "dx": dx, "dy": dy,
                           "x": round(x, 8), "y": round(y, 8), "seed": seed + repeat,
                           "baseline": dx == 0 and dy == 0})
    expert = Path(expert).resolve()
    # Dependency fingerprints prevent accidentally resuming with different physics/code.
    dependencies = set(expert.parent.glob("*.py"))
    for folder in (ROOT / "scripts/metrics", ROOT / "scripts/learning"):
        dependencies.update(folder.glob("*.py"))
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(dependencies) if path.resolve() != Path(__file__).resolve()}
    return {"version": 1, "expert": str(expert), "sources": sources,
            "python": sys.executable, "center": [center_x, center_y], "offsets": offsets,
            "repeats": repeats, "retries": retries, "yaw_deg": 0., "pos_jitter": 0.,
            "params": params, "trials": trials, "video": bool(video)}


def comparable_plan(plan):
    # Older plans fingerprinted this orchestration file too. UI/logging changes
    # must not invalidate physics results; expert dependencies remain strict.
    plan = json.loads(json.dumps(plan))
    plan["sources"].pop(str(Path(__file__).resolve()), None)
    plan.setdefault("video", False)
    return plan


def command_for(plan, trial, output):
    command = [plan["python"], "-X", "utf8", "-u", plan["expert"],
        "--grasp_mode", "physical", "--stop_after", "fold", "--cloth_yaw_deg", "0",
        "--pos_jitter", "0", "--cloth_x", str(trial["x"]), "--cloth_y", str(trial["y"]),
        "--seed", str(trial["seed"]), "--retries", str(plan["retries"]),
        "--no_gui", "--video_dir", str(Path(output) / "artifacts" / trial["id"])]
    command += ["--save_all_videos"] if plan.get("video", False) else ["--no_video"]
    for key, value in sorted(plan["params"].items()):
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend((f"--{key}", str(value)))
    return command + ["presets=isaacsim_physx"]


def parse_result(log, trial, *, threshold=.05, iou_min=.8):
    result = None
    for line in log.splitlines():
        if line.startswith("[fold] RESULT "):
            result = json.loads(line[len("[fold] RESULT "):])
    if not isinstance(result, dict):
        raise ValueError("No expert RESULT record")
    start = result.get("cloth_start")
    if (result.get("grasp_mode") != "physical" or result.get("stop_after") != "fold"
            or not isinstance(start, list) or len(start) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in start)
            or any(abs(a - b) > 1e-5 for a, b in zip(start, (trial["x"], trial["y"])))):
        raise ValueError("Expert result does not match the requested physical full-fold position")
    if not isinstance(result.get("success"), bool):
        raise ValueError("Expert success flag is missing")
    distances = result.get("final_distances_m")
    if result["success"] and (result.get("status") != "completed" or not isinstance(distances, list)
            or len(distances) != 2 or not all(isinstance(d, (int, float)) and math.isfinite(d)
                                             and 0 <= d <= threshold for d in distances)):
        raise ValueError("Success is not supported by both final corner distances")
    if result.get("status") == "completed":
        iou = (result.get("shape_metrics") or {}).get("iou")
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError("Completed trial has no valid IoU metric; wrong expert version?")
        if result["success"] and iou < iou_min:
            raise ValueError("Success is not supported by the required IoU")
    return result


def stop_process(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_trial(plan, trial, output, timeout):
    output = Path(output)
    (output / "logs").mkdir(exist_ok=True)
    log_path = output / "logs" / f"{trial['id']}.log"
    row = {**trial, "success": False, "process_status": "OK"}
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command_for(plan, trial, output), stdout=log, stderr=subprocess.STDOUT,
            cwd=ROOT, start_new_session=os.name != "nt")
        try:
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                try:
                    code = process.wait(timeout=min(15., remaining))
                    break
                except subprocess.TimeoutExpired:
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    meaningful = [line for line in tail if line.startswith(("[fold]", "[virtual_grasp]", "[INFO]"))]
                    message = meaningful[-1] if meaningful else (tail[-1] if tail else "starting simulator")
                    print(f"[xy] {trial['id']} elapsed={time.monotonic() - start:.0f}s {message[:220]}", flush=True)
            if code != 0:
                row.update(process_status="ERROR", returncode=code)
        except subprocess.TimeoutExpired:
            stop_process(process)
            row["process_status"] = "TIMEOUT"
        except BaseException:
            stop_process(process)
            raise
    try:
        result = parse_result(log_path.read_text(encoding="utf-8", errors="replace"), trial,
                              threshold=plan["params"]["success_distance"],
                              iou_min=plan["params"]["shape_iou_threshold"])
        row.update(result_status=result.get("status"), phase=result.get("phase"),
                   distances_m=result.get("final_distances_m"), grasps=result.get("grasps"),
                   run_dir=result.get("run_dir"), max_drift_m=result.get("max_drift_m"))
        row.update(iou=(result.get("shape_metrics") or {}).get("iou"),
                   corner_success=result.get("corner_success"), shape_success=result.get("shape_success"))
        row["success"] = row["process_status"] == "OK" and result["success"]
        if result.get("status") == "error":
            row["process_status"] = "ERROR"
        if plan.get("video", False):
            row["video_files"] = [str(path) for path in Path(result["run_dir"]).glob("*.mp4") if path.stat().st_size > 0]
            if not row["video_files"] and row["process_status"] == "OK":
                row.update(process_status="VIDEO_MISSING", success=False)
    except (ValueError, TypeError) as exc:
        row["result_error"] = str(exc)
        if row["process_status"] == "OK":
            row["process_status"] = "INVALID_RESULT"
    row.update(duration_s=round(time.monotonic() - start, 2), log=str(log_path))
    return row


def summarize(plan, results, *, min_rate=.8):
    points = []
    for point in sorted({t["point"] for t in plan["trials"]}):
        trials = [t for t in plan["trials"] if t["point"] == point]
        rows = [results[t["id"]] for t in trials if t["id"] in results]
        successes = sum(r["success"] for r in rows)
        rate = successes / len(rows) if rows else None
        points.append({"point": point, "x": trials[0]["x"], "y": trials[0]["y"],
            "planned": len(trials), "attempted": len(rows), "successes": successes,
            "errors": sum(r["process_status"] != "OK" for r in rows), "success_rate": rate,
            "mean_iou": (sum(r["iou"] for r in rows if isinstance(r.get("iou"), (int, float))) /
                         sum(isinstance(r.get("iou"), (int, float)) for r in rows)
                         if any(isinstance(r.get("iou"), (int, float)) for r in rows) else None),
            "candidate": len(rows) == len(trials) and rate >= min_rate})
    return {"yaw_deg": 0, "min_rate": min_rate,
            "success_distance_m": plan["params"]["success_distance"],
            "shape_iou_threshold": plan["params"]["shape_iou_threshold"], "points": points,
            "aligned_low_level_bc_ready": False,
            "note": "Passing discrete points are preliminary candidates, not proof of a continuous reachable rectangle or BC performance."}


def save_summary(plan, results, output, min_rate):
    summary = summarize(plan, results, min_rate=min_rate)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["points"][0]))
        writer.writeheader()
        writer.writerows(summary["points"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/physical-xy-probe")
    parser.add_argument("--center_x", type=float, default=.5)
    parser.add_argument("--center_y", type=float, default=-.05)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-.03, 0., .03])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--params", type=Path, help="JSON of fixed expert settings; no XY/yaw/noise overrides")
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--min_rate", type=float, default=.8)
    parser.add_argument("--max_trials", type=int, help="Execute only this many pending trials, then stop")
    parser.add_argument("--run", action="store_true", help="Execute the saved plan; otherwise plan only")
    parser.add_argument("--video", action="store_true", help="Save MP4 for both successful and failed attempts (use a new --out)")
    parser.add_argument("--iou_min", type=float, default=.8,
                        help="Per-fold IoU gate (.8 default); 0 reproduces the reference test command")
    args = parser.parse_args()
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 < args.min_rate <= 1:
            raise ValueError("Invalid timeout or min_rate")
        if args.max_trials is not None and args.max_trials < 1:
            raise ValueError("max_trials must be positive")
        params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
        plan = make_plan(center_x=args.center_x, center_y=args.center_y, offsets=args.offsets,
                         repeats=args.repeats, seed=args.seed, retries=args.retries, params=params,
                         video=args.video, iou_min=args.iou_min)
        plan["timeout"], plan["min_rate"] = args.timeout, args.min_rate
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=True)
        plan_path = args.out / "plan.json"
        if plan_path.exists() and comparable_plan(json.loads(plan_path.read_text(encoding="utf-8"))) != comparable_plan(plan):
            raise ValueError("Plan/code differs from this output directory. Use a new --out directory.")
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(f"[xy] yaw=0, noise=0, video={plan['video']}; {len(plan['trials'])} trials; plan: {plan_path}", flush=True)
    if not args.run:
        print("[xy] Plan only. Add --run to execute. Existing results are resumed automatically.")
        return
    results_path = args.out / "results.jsonl"
    try:
        with (args.out / ".running.lock").open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError:
        parser.error(f"A run lock already exists: {args.out / '.running.lock'}. Check for a running/stopped probe.")
    try:
        results = {row["id"]: row for row in (json.loads(line) for line in
                   results_path.read_text(encoding="utf-8").splitlines() if line.strip())} if results_path.exists() else {}
        executed = 0
        for trial in plan["trials"]:
            if trial["id"] in results:
                continue
            baseline = [t for t in plan["trials"] if t["baseline"]]
            if not trial["baseline"] and not any(results[t["id"]]["success"] for t in baseline):
                print("[xy] Baseline failed every repeat; stop and improve the expert first.")
                break
            if args.max_trials is not None and executed >= args.max_trials:
                break
            print(f"[xy] {trial['id']} x={trial['x']:.3f} y={trial['y']:.3f} seed={trial['seed']}", flush=True)
            row = run_trial(plan, trial, args.out, args.timeout)
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            results[trial["id"]] = row
            executed += 1
            save_summary(plan, results, args.out, args.min_rate)
            print(f"[xy] {row['process_status']} success={row['success']}", flush=True)
            if row["process_status"] != "OK":
                print("[xy] Infrastructure/result error; stop and inspect the trial log.")
                break
        save_summary(plan, results, args.out, args.min_rate)
        print(f"[xy] Summary: {args.out / 'summary.csv'}", flush=True)
    finally:
        (args.out / ".running.lock").unlink()


if __name__ == "__main__":
    main()
