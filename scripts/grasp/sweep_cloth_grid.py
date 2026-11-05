"""
Cloth settings grid for virtual/physical unification (experiment 009).

Runs rectangular_fold_demo.py with --grasp_mode physical --stop_after transport, so every
run grasps a corner, checks the grasp, carries the corner along the fold arc and places it
on its goal corner. A run succeeds when the grasp check passed, the corner was not dropped
on the way and it landed within 2 cm of its goal.
  probe : build the cloth once per mesh refinement and record the real vertex count
  A     : bend stiffness (10 values, log spaced) x up to 4 meshes picked from the probe
  B     : on one mesh, 4 more bend values between each pair where the result flips
  R     : best 3 settings repeated 5 times with the cloth shifted by up to 3 cm
  F     : the best setting folded completely (both corners), 3 times
Each run is its own process, results go to a CSV right after each run, and a stopped
sweep resumes where it left off.

Run from the Isaac Lab folder:
    .\\isaaclab.bat -p C:\\cloth-folding-rl\\scripts\\grasp\\sweep_cloth_grid.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Never crash on a console that cannot show Korean text
sys.stdout.reconfigure(errors="replace")

REPO = Path(__file__).resolve().parents[2]
FOLD_SCRIPT = Path(__file__).resolve().parent / "rectangular_fold_demo.py"

# Unified cloth (team table, 2026-11) and the physical grasp settings from 008. Same for every run.
FIXED = {
    "grasp_mode": "physical",
    "cloth_size": 0.30,
    "cloth_thickness": 0.003,
    "cloth_density": 180.0,
    "youngs_modulus": 1e6,
    "poissons_ratio": 0.25,
    "damping": 0.25,
    "cloth_friction": 0.5,    # between table (0.3) and finger (1.5) so both combine modes act as intended
    "finger_friction": 1.5,
    "table_friction": 0.3,
    "table_length": 1.5,
    "table_width": 0.9,
    "cloth_x": 0.50,
    "cloth_y": -0.05,
    "open_width": 0.04,
    "press_depth": 0.0,
}

# Bend stiffness: 10 values from 185 to 10000, each about 1.56x the previous one
BEND_MIN, BEND_MAX, BEND_COUNT = 185.0, 10000.0, 10
BEND_VALUES = [round(BEND_MIN * (BEND_MAX / BEND_MIN) ** (i / (BEND_COUNT - 1)), 1) for i in range(BEND_COUNT)]
# Mesh: try these refinement factors, then keep the ones closest to the target vertex spacings
REFINE_CANDIDATES = [16, 24, 32, 48, 66.67, 96, 128]
TARGET_SPACINGS_MM = [9.4, 6.0, 4.5, 3.0]
BOUNDARY_POINTS = 4   # stage B: bend values inserted between a passing and a failing neighbour
BOUNDARY_MAX = 10

PARAM_KEYS = [*FIXED, "bend_stiffness", "mesh_refinement", "seed", "pos_jitter", "stop_after"]
RESULT_KEYS = [
    "result_status", "success", "fail_phase", "mesh_node_count", "spacing_mm",
    "grasp_ok", "grasp_attempts", "open_gap_above_mm", "open_gap_mm", "rise_mm", "corner_tcp_mm",
    "max_drift_mm", "slip_phase", "place_error_mm", "lower_error_mm", "applied_size_mm", "applied_bend", "run_dir",
]
CSV_FIELDS = ["run_id", "stage", "status", "duration_s", *PARAM_KEYS, *RESULT_KEYS]

parser = argparse.ArgumentParser(description="Cloth unification grid (physical grasp + transport)")
parser.add_argument("--out", type=Path, default=REPO / "logs" / "sweep_009", help="Folder for results.csv and run logs")
parser.add_argument("--timeout", type=float, default=1800.0, help="Seconds before a stuck run is killed")
parser.add_argument("--retry_failed", action="store_true", help="Run ERROR/TIMEOUT runs again")
parser.add_argument("--min_spacing_mm", type=float, default=4.0,
                    help="Skip meshes finer than this (finer ones overflowed the PhysX contact buffer)")
parser.add_argument("--top", type=int, default=3, help="Stage R: how many settings to repeat")
parser.add_argument("--repeats", type=int, default=5, help="Stage R: runs per setting (seeds 1..N)")
parser.add_argument("--jitter", type=float, default=0.03, help="Stage R/F: max cloth start shift in x and y [m]")
parser.add_argument("--need", type=int, default=4, help="Stage R: successes needed out of --repeats")
parser.add_argument("--folds", type=int, default=3, help="Stage F: full folds of the best setting")
parser.add_argument("--video_stages", default="all",
                    help="Stages recorded to videos/: 'all' (A,B,R,F) or e.g. 'R,F'. '' = none. Probe runs are never recorded")
parser.add_argument("--all_videos", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep the video of failed runs too (--no-all_videos: successful runs only)")
parser.add_argument("--no_gui", action="store_true", help="Run without the Isaac Sim window (default: window shown)")
parser.add_argument("--video_dir", type=Path, default=REPO / "videos" / "sweep_009")
args = parser.parse_args()


def run_id_of(params: dict) -> str:
    if params["stop_after"] == "probe":
        return f"probe_r{params['mesh_refinement']:g}"
    tag = "fold_" if params["stop_after"] == "fold" else ""
    return f"{tag}b{params['bend_stiffness']:g}_r{params['mesh_refinement']:g}_s{params['seed']}"


def load_done(csv_path: Path) -> dict[str, dict]:
    """Rows already in the CSV, keyed by run_id (the last row wins)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def command_for(params: dict, stage: str, trial_dir: Path) -> list[str]:
    cmd = [sys.executable, str(FOLD_SCRIPT)]
    for key in PARAM_KEYS:
        cmd += [f"--{key}", str(params[key])]
    gui = ["--no_gui"] if args.no_gui else []
    stages = ["A", "B", "R", "F"] if args.video_stages == "all" else args.video_stages.split(",")
    if stage in stages:
        # One folder per run, so a video can be found from its CSV row (run_dir column)
        video = ["--save_all_videos"] if args.all_videos else []
        return cmd + gui + video + ["--video_dir", str(args.video_dir / f"{stage}_{run_id_of(params)}"),
                                    "presets=isaacsim_physx"]
    return cmd + gui + ["--no_video", "--video_dir", str(trial_dir), "presets=isaacsim_physx"]


def flatten(result: dict) -> dict:
    """Turn the demo's RESULT line into CSV columns (lengths in mm)."""
    grasps = result.get("grasps") or []
    first = grasps[0] if grasps else {}
    distances = result.get("final_distances_m") or []
    mm = lambda v: None if v is None else round(v * 1000, 2)
    applied = result.get("applied") or {}
    return {
        "result_status": result.get("status"), "success": result.get("success"),
        "fail_phase": result.get("phase"), "mesh_node_count": result.get("mesh_node_count"),
        "spacing_mm": None if result.get("spacing_mm") is None else round(result["spacing_mm"], 3),
        "grasp_ok": first.get("ok"), "grasp_attempts": first.get("attempts"),
        "open_gap_above_mm": mm(first.get("open_gap_above_m")), "lower_error_mm": mm(first.get("lower_error_m")),
        "open_gap_mm": mm(first.get("open_gap_m")), "rise_mm": mm(first.get("rise_m")),
        "corner_tcp_mm": mm(first.get("corner_tcp_m")), "max_drift_mm": mm(result.get("max_drift_m")),
        "slip_phase": result.get("slip_phase"), "place_error_mm": mm(distances[0]) if distances else None,
        "applied_size_mm": mm(applied.get("cloth_size")), "applied_bend": applied.get("bend_stiffness"),
        "run_dir": result.get("run_dir"),
    }


def run_one(params: dict, stage: str, log_dir: Path) -> dict:
    """Run one setting in a new process and return a CSV row."""
    run_id = run_id_of(params)
    log_path = log_dir / f"{run_id}.log"
    start = time.time()
    status = "ERROR"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run(command_for(params, stage, log_dir / "trials"), stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "OK" if proc.returncode == 0 else "ERROR"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"

    row = {"run_id": run_id, "stage": stage, "status": status, "duration_s": f"{time.time() - start:.0f}",
           **{k: params[k] for k in PARAM_KEYS}}
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[fold] RESULT "):
            result = json.loads(line[len("[fold] RESULT "):])
    if result is None:
        row["status"] = "NO_RESULT" if status == "OK" else status
    else:
        row.update(flatten(result))
    return row


def append_row(csv_path: Path, row: dict) -> None:
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != CSV_FIELDS:   # written by an older version: keep it aside, start a new table
            old = csv_path.with_name(csv_path.stem + "_old.csv")
            csv_path.replace(old)
            print(f"[sweep] 열 구성이 바뀌어 기존 결과를 {old.name}으로 옮겼습니다")
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def succeeded(row: dict) -> bool:
    """Grasp check passed, not dropped while carried, corner placed within 2 cm of its goal."""
    return row["status"] == "OK" and row["success"] == "True"


def setting(bend: float, mesh: float, seed: int = 0, jitter: float = 0.0, stop_after: str = "transport") -> dict:
    return {**FIXED, "bend_stiffness": bend, "mesh_refinement": mesh, "seed": seed,
            "pos_jitter": jitter, "stop_after": stop_after}


def spacing_of(rows: list[dict], mesh: float) -> float:
    """Measured vertex spacing [mm] of a mesh refinement (inf if never measured)."""
    values = [float(r["spacing_mm"]) for r in rows if float(r["mesh_refinement"]) == mesh and r["spacing_mm"]]
    return values[0] if values else float("inf")


def pattern_of(rows: list[dict], mesh: float) -> list[bool | None]:
    """Success over BEND_VALUES on one mesh (None = not run)."""
    runs = {float(r["bend_stiffness"]): succeeded(r) for r in rows
            if float(r["mesh_refinement"]) == mesh and r["seed"] == "0" and r["stage"] in ("A", "B")}
    return [runs.get(b) for b in BEND_VALUES]


def pick_meshes(probe_rows: list[dict]) -> list[float]:
    """Refinements whose real vertex spacing is closest to each target, without duplicates."""
    # Finer meshes overflow the PhysX deformable contact buffer (seen at 2.34 mm / 16641 nodes)
    ok = [r for r in probe_rows if r["status"] == "OK" and r["spacing_mm"]
          and float(r["spacing_mm"]) >= args.min_spacing_mm]
    picked, seen_nodes = [], set()
    for target in TARGET_SPACINGS_MM:
        best = min(ok, key=lambda r: abs(float(r["spacing_mm"]) - target), default=None)
        if best is None or best["mesh_node_count"] in seen_nodes:
            continue
        seen_nodes.add(best["mesh_node_count"])
        picked.append(float(best["mesh_refinement"]))
    return picked


def pick_boundary(rows: list[dict], meshes: list[float]) -> tuple[float | None, list[float]]:
    """One mesh for stage B and the bend values to add there.

    The mesh is the coarsest one whose pattern over the bend values equals the finest
    mesh's (results stopped changing); otherwise the finest. New bend values go between
    neighbours where the result flips.
    """
    if not meshes:
        return None, []
    by_spacing = sorted(meshes, key=lambda m: spacing_of(rows, m))   # finest first
    finest = by_spacing[0]
    chosen = next(m for m in reversed(by_spacing) if pattern_of(rows, m) == pattern_of(rows, finest))
    pattern = pattern_of(rows, chosen)
    extra = []
    for i in range(len(BEND_VALUES) - 1):
        if pattern[i] is None or pattern[i + 1] is None or pattern[i] == pattern[i + 1]:
            continue
        low, high = BEND_VALUES[i], BEND_VALUES[i + 1]
        ratio = (high / low) ** (1 / (BOUNDARY_POINTS + 1))
        extra += [round(low * ratio ** k, 1) for k in range(1, BOUNDARY_POINTS + 1)]
    return chosen, extra[:BOUNDARY_MAX]


def rank(rows: list[dict]) -> list[dict]:
    """Single-try settings, best first: success, then smallest place error, then least slip."""
    ok = [r for r in rows if r["status"] == "OK" and r["seed"] == "0" and r["stage"] in ("A", "B")
          and r["grasp_ok"] == "True"]
    big = 1e9
    return sorted(ok, key=lambda r: (not succeeded(r), float(r["place_error_mm"] or big),
                                     float(r["max_drift_mm"] or big)))


def check_preflight(row: dict) -> list[str]:
    """Problems that would make the whole sweep meaningless."""
    if row["status"] != "OK":
        return [f"실행 상태가 {row['status']}입니다 (로그 파일 확인)"]
    problems = []
    if row["applied_size_mm"] and abs(float(row["applied_size_mm"]) - FIXED["cloth_size"] * 1000) > 1.0:
        problems.append(f"천 크기가 {row['applied_size_mm']} mm로 적용됐습니다. {FIXED['cloth_size'] * 1000:.0f} mm여야 합니다")
    if row["applied_bend"] and abs(float(row["applied_bend"]) - float(row["bend_stiffness"])) > 1e-3:
        problems.append(f"굽힘 강성이 {row['applied_bend']}로 적용됐습니다. 요청값은 {row['bend_stiffness']}입니다")
    # Checked in free air above the cloth; just before closing the fingers may already touch the cloth
    if row["open_gap_above_mm"] and abs(float(row["open_gap_above_mm"]) - FIXED["open_width"] * 1000) > 2.0:
        problems.append(f"천 위 공중에서 손가락 간격이 {row['open_gap_above_mm']} mm입니다. {FIXED['open_width'] * 1000:.0f} mm여야 합니다")
    return problems


def print_stage_a(rows: list[dict], meshes: list[float]) -> None:
    print("\n[sweep] A 결과 (O = 집어서 옮겨 놓은 뒤 모서리가 목표 2 cm 이내, x = 실패, - = 없음)")
    print("  굽힘강성 " + "".join(f"| 간격 {spacing_of(rows, m):4.1f} mm " for m in meshes))
    patterns = [pattern_of(rows, m) for m in meshes]
    for i, bend in enumerate(BEND_VALUES):
        cells = ["-" if p[i] is None else ("O" if p[i] else "x") for p in patterns]
        print(f"  {bend:>8g} " + "".join(f"|      {c}       " for c in cells))


def print_failures(rows: list[dict]) -> None:
    """How the failed transport runs ended (grasp check / dropped / missed the goal)."""
    failed = [r for r in rows if r["stage"] in ("A", "B", "R") and r["status"] == "OK" and not succeeded(r)]
    if not failed:
        return
    kinds = {}
    for r in failed:
        kind = r["result_status"] if r["result_status"] != "completed" else "목표 2 cm 밖"
        kinds[kind] = kinds.get(kind, 0) + 1
    print("[sweep] 실패 유형: " + ", ".join(f"{k} {v}회" for k, v in sorted(kinds.items())))


def print_repeat_summary(rows: list[dict]) -> None:
    repeat_rows = [r for r in rows if r["stage"] == "R"]
    if not repeat_rows:
        return
    print(f"\n[sweep] R 반복 결과 ({args.repeats}번 중 {args.need}번 이상이면 통과, 천 위치 ±{args.jitter * 1000:.0f} mm)")
    for bend, mesh in sorted({(r["bend_stiffness"], r["mesh_refinement"]) for r in repeat_rows}):
        runs = [r for r in repeat_rows if (r["bend_stiffness"], r["mesh_refinement"]) == (bend, mesh) and r["status"] == "OK"]
        good = sum(succeeded(r) for r in runs)
        spacing = runs[0]["spacing_mm"] if runs else "?"
        print(f"  굽힘강성 {bend} · 정점간격 {spacing} mm : {good}/{len(runs)} | {'통과' if good >= args.need else '불통과'}")


def main() -> None:
    log_dir = args.out / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"
    skip_status = {"OK", "NO_RESULT"} if args.retry_failed else {"OK", "NO_RESULT", "ERROR", "TIMEOUT"}

    def run_all(runs: list[dict], stage: str) -> None:
        done = load_done(csv_path)
        todo = [p for p in runs if done.get(run_id_of(p), {}).get("status") not in skip_status]
        for i, params in enumerate(todo, start=1):
            print(f"[sweep] {stage} {i}/{len(todo)} {run_id_of(params)}", flush=True)
            row = run_one(params, stage, log_dir)
            append_row(csv_path, row)
            print(f"[sweep]   -> {row['status']} / {row.get('result_status')} ({row['duration_s']}초)", flush=True)

    def rows_now() -> list[dict]:
        return list(load_done(csv_path).values())

    # probe: real vertex count per refinement (refinement is not "vertices per edge")
    print(f"[sweep] 메시 확인: refinement {REFINE_CANDIDATES} 각각 천만 만들어 정점 수를 잽니다")
    run_all([setting(BEND_VALUES[0], float(r), stop_after="probe") for r in REFINE_CANDIDATES], "probe")
    probe_rows = [r for r in rows_now() if r["stage"] == "probe"]
    for r in sorted(probe_rows, key=lambda r: float(r["mesh_refinement"])):
        print(f"  refinement {r['mesh_refinement']:>6} -> 정점 {r['mesh_node_count']}개, 간격 {r['spacing_mm']} mm ({r['status']})")
    meshes = pick_meshes(probe_rows)
    if not meshes:
        print("[sweep] 메시 확인이 모두 실패했습니다. runs 폴더의 probe 로그를 확인해 주세요.")
        sys.exit(1)
    print(f"[sweep] 고른 메시(refinement): {meshes}")
    grid = [setting(b, m) for m in meshes for b in BEND_VALUES]

    # preflight: the finest mesh, so its duration is an upper bound for the time estimate
    preflight = grid[-1]
    if run_id_of(preflight) not in load_done(csv_path):
        print(f"[sweep] 점검 실행 중: {run_id_of(preflight)} ... (통과 메시지가 나올 때까지 지켜봐 주세요)")
        row = run_one(preflight, "A", log_dir)
        problems = check_preflight(row)
        if problems:
            print("[sweep] 점검 실패. 전체 실행을 시작하지 않고 멈춥니다:")
            for problem in problems:
                print(f"  - {problem}")
            print(f"[sweep] 로그 파일: {log_dir / (row['run_id'] + '.log')}")
            sys.exit(1)
        append_row(csv_path, row)
        remaining = len(grid) - 1 + BOUNDARY_MAX + args.top * args.repeats + args.folds * 2
        hours = remaining * float(row["duration_s"]) / 3600
        print(f"[sweep] 점검 통과 (1회 {row['duration_s']}초, 가장 촘촘한 메시 기준, 결과 {row['result_status']}).")
        print(f"[sweep] 이제 자리를 비워도 됩니다. 남은 실행 최대 약 {remaining}회, 최대 약 {hours:.1f}시간 예상.")

    run_all(grid, "A")
    print_stage_a(rows_now(), meshes)

    chosen, extra = pick_boundary(rows_now(), meshes)
    if chosen is not None and extra:
        print(f"\n[sweep] B: refinement {chosen:g}에서 결과가 바뀌는 구간에 굽힘 강성 {extra} 추가")
        run_all([setting(b, chosen) for b in extra], "B")
    else:
        print("\n[sweep] B: 결과가 바뀌는 구간이 없어 건너뜁니다 (전부 성공 또는 전부 실패)")

    ranked = [r for r in rank(rows_now()) if chosen is None or float(r["mesh_refinement"]) == chosen]
    top = ranked[: args.top]
    if not top:
        print("\n[sweep] 집기 확인을 통과한 설정이 없어 R, F는 건너뜁니다.")
    else:
        print(f"\n[sweep] R: 상위 {len(top)}개 설정을 {args.repeats}번씩 (천 위치 ±{args.jitter * 1000:.0f} mm)")
        run_all([setting(float(r["bend_stiffness"]), float(r["mesh_refinement"]), seed, args.jitter)
                 for r in top for seed in range(1, args.repeats + 1)], "R")
        print_repeat_summary(rows_now())
        best = top[0]
        print(f"\n[sweep] F: 1위 설정(굽힘강성 {best['bend_stiffness']})으로 접기 전체 {args.folds}번")
        run_all([setting(float(best["bend_stiffness"]), float(best["mesh_refinement"]), seed, args.jitter, "fold")
                 for seed in range(1, args.folds + 1)], "F")
        folds = [r for r in rows_now() if r["stage"] == "F" and r["status"] == "OK"]
        print(f"[sweep] F 결과: 접기 성공 {sum(succeeded(r) for r in folds)}/{len(folds)}")
    print_failures(rows_now())
    print(f"[sweep] 전체 결과 표: {csv_path}")


if __name__ == "__main__":
    main()
