"""
Run press_close_grasp.py over a grid of settings (experiment 008).
Stage 1 runs every combination once. Stage 2 then picks the best combinations
automatically and repeats each one with the cloth shifted slightly (#14: 4 of 5).
Each run is its own process. Results are appended to a CSV right after each run,
so the sweep can be stopped and resumed.

Run from the Isaac Lab folder:
    .\\isaaclab.bat -p C:\\cloth-folding-rl\\scripts\\grasp\\sweep_press_close.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

# Never crash on a console that cannot show Korean text
sys.stdout.reconfigure(errors="replace")

REPO = Path(__file__).resolve().parents[2]
GRASP_SCRIPT = Path(__file__).resolve().parent / "press_close_grasp.py"

# Grid (lengths in m). Every combination of these values is run. Edit here to change the sweep.
GRID = {
    "open_width": [0.03, 0.04],
    "edge_refinement": [32, 48, 64],
    "table_friction": [0.03, 0.3],
}
# Same for every run. Finger-cloth friction (1.5) is fixed inside press_close_grasp.py
FIXED = {
    "press_depth": 0.0,
    "thickness": 0.003,
    "bend_scale": 1.0,
    "grasp_layout": "axis",
    "seed": 0,
    "pos_jitter": 0.0,
}

# The first run checks that the gripper opening and mesh settings are applied.
# It uses the finest mesh, so its duration is a safe upper bound for the time estimate
PREFLIGHT = {**FIXED, "open_width": 0.03, "edge_refinement": 64, "table_friction": 0.03}

PARAM_KEYS = [*GRID, *FIXED]
RESULT_KEYS = [
    "num_nodes", "settle_gap", "cloth_dx", "cloth_dy", "inset", "open_gap", "left_in", "right_in",
    "press_gap", "ridge", "finger_gap", "gathered", "hold_min_rise", "corner_rise",
    "corner_tcp", "corner_tcp_max", "passed", "quality_ok",
]
CSV_FIELDS = ["run_id", "status", "duration_s", *PARAM_KEYS, *RESULT_KEYS]

parser = argparse.ArgumentParser(description="Grid sweep for the press-and-close grasp")
parser.add_argument("--out", type=Path, default=REPO / "logs" / "sweep_008", help="Folder for results.csv and run logs")
parser.add_argument("--timeout", type=float, default=900.0, help="Seconds before a stuck run is killed")
parser.add_argument("--retry_failed", action="store_true", help="Run ERROR/TIMEOUT combinations again")
parser.add_argument("--skip_preflight", action="store_true")
parser.add_argument("--top", type=int, default=3, help="Stage 2: how many stage-1 combinations to repeat")
parser.add_argument("--repeats", type=int, default=5, help="Stage 2: runs per combination (seeds 1..N)")
parser.add_argument("--jitter", type=float, default=0.01, help="Stage 2: max cloth start shift in x and y [m]")
parser.add_argument("--need", type=int, default=4, help="Stage 2: successes needed out of --repeats (#14)")
args = parser.parse_args()


def run_id_of(params: dict) -> str:
    return (
        f"w{params['open_width'] * 1000:.0f}_r{params['edge_refinement']}"
        f"_tf{params['table_friction']:g}_s{params['seed']}"
    )


def load_done(csv_path: Path) -> dict[str, dict]:
    """Rows already in the CSV, keyed by run_id (the last row wins)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def run_one(params: dict, log_dir: Path) -> dict:
    """Run one combination in a new process and return a CSV row."""
    run_id = run_id_of(params)
    cmd = [sys.executable, str(GRASP_SCRIPT)]
    for key in PARAM_KEYS:
        cmd += [f"--{key}", str(params[key])]
    cmd += ["--no_gui", "physics=physx"]
    log_path = log_dir / f"{run_id}.log"
    start = time.time()
    status = "ERROR"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "OK" if proc.returncode == 0 else "ERROR"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"

    row = {"run_id": run_id, "status": status, "duration_s": f"{time.time() - start:.0f}", **params}
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[press_close] RESULT "):
            result = json.loads(line[len("[press_close] RESULT "):])
    if result is None:
        if status == "OK":
            status = "NO_RESULT"
        row["status"] = status
    else:
        row.update({key: result.get(key) for key in RESULT_KEYS})
    return row


def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def check_preflight(row: dict) -> list[str]:
    """Problems that would make the whole sweep meaningless."""
    if row["status"] != "OK":
        return [f"실행 상태가 {row['status']}입니다 (로그 파일 확인)"]
    problems = []
    expected_nodes = (PREFLIGHT["edge_refinement"] + 1) ** 2
    if int(row["num_nodes"]) != expected_nodes:
        problems.append(f"천 정점 수가 {row['num_nodes']}개입니다. {expected_nodes}개여야 합니다 (메시 해상도 설정 미적용)")
    requested = PREFLIGHT["open_width"] * 1000
    if abs(float(row["open_gap"]) - requested) > 2.0:
        problems.append(f"열린 손가락 간격이 {float(row['open_gap']):.1f} mm입니다. 요청값은 {requested:.1f} mm (열림 간격 설정 미적용)")
    return problems


def combo_id(run_id: str) -> str:
    """Run id without the seed, e.g. w30_r64_tf0.3"""
    return run_id.rsplit("_s", 1)[0]


def pick_repeats(stage1_rows: list[dict]) -> list[dict]:
    """Stage-1 combinations worth repeating: lifted, fingers on the cloth, best quality first."""
    candidates = [
        r for r in stage1_rows
        if r["status"] == "OK" and r["passed"] == "True"
        and float(r["left_in"]) > 0 and float(r["right_in"]) > 0
    ]
    candidates.sort(key=lambda r: (r["quality_ok"] != "True", float(r["corner_tcp"])))
    repeats = []
    for r in candidates[: args.top]:
        base = {**FIXED, "open_width": float(r["open_width"]), "edge_refinement": int(r["edge_refinement"]),
                "table_friction": float(r["table_friction"]), "pos_jitter": args.jitter}
        repeats += [{**base, "seed": seed} for seed in range(1, args.repeats + 1)]
    return repeats


def print_repeat_summary(csv_path: Path) -> None:
    rows = [r for r in load_done(csv_path).values() if r["seed"] != "0"]
    if not rows:
        return
    print(f"\n[sweep] 2차 반복 결과 ({args.repeats}번 중 {args.need}번 이상이면 #14 통과)")
    for combo in sorted({combo_id(r["run_id"]) for r in rows}):
        runs = [r for r in rows if combo_id(r["run_id"]) == combo and r["status"] == "OK"]
        lifted = sum(r["passed"] == "True" for r in runs)
        both = sum(r["passed"] == "True" and r["quality_ok"] == "True" for r in runs)
        verdict = "통과" if both >= args.need else "불통과"
        print(f"  {combo:<16} 들림 {lifted}/{len(runs)} | 들림+품질 {both}/{len(runs)} | {verdict}")


def print_summary(csv_path: Path) -> None:
    rows = [r for r in load_done(csv_path).values() if r["seed"] == "0"]
    ok = [r for r in rows if r["status"] == "OK"]
    print(f"\n[sweep] 1차 완료: 총 {len(rows)}회, 정상 {len(ok)}회, 오류·시간초과 {len(rows) - len(ok)}회")
    # Rank: lifted first, then good quality, then the corner closest to the gripper
    ok.sort(key=lambda r: (r["passed"] != "True", r["quality_ok"] != "True", float(r["corner_tcp"])))
    print("[sweep] 상위 조합 (단위 mm. 들림=20 mm 이상 유지, 품질=모서리가 그리퍼에서 3 cm 이내)")
    for r in ok[:12]:
        on_cloth = float(r["left_in"]) > 0 and float(r["right_in"]) > 0
        print(
            f"  {r['run_id']:<22} 들림 {'O' if r['passed'] == 'True' else 'X'}"
            f" | 품질 {'O' if r['quality_ok'] == 'True' else 'X'}"
            f" | 모서리거리={float(r['corner_tcp']):.1f} | 빨려든폭={float(r['gathered']):.1f}"
            f" | 유지높이={float(r['hold_min_rise']):.1f} | 봉우리={float(r['ridge']):.1f}"
            f" | 천높이={float(r['settle_gap']):.1f} | 손가락_천위={'예' if on_cloth else '아니오'}"
        )
    print(f"[sweep] 전체 결과 표: {csv_path}")


def main() -> None:
    log_dir = args.out / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"

    grid = [{**FIXED, **dict(zip(GRID, values))} for values in itertools.product(*GRID.values())]
    skip_status = {"OK", "NO_RESULT"} if args.retry_failed else {"OK", "NO_RESULT", "ERROR", "TIMEOUT"}

    def pending(runs: list[dict]) -> list[dict]:
        done = load_done(csv_path)
        return [p for p in runs if done.get(run_id_of(p), {}).get("status") not in skip_status]

    if not args.skip_preflight and run_id_of(PREFLIGHT) not in load_done(csv_path):
        print(f"[sweep] 점검 실행 중: {run_id_of(PREFLIGHT)} ... (통과 메시지가 나올 때까지 지켜봐 주세요)")
        row = run_one(PREFLIGHT, log_dir)
        problems = check_preflight(row)
        if problems:
            # Not saved to the CSV, so the preflight runs again after a fix
            print("[sweep] 점검 실패. 전체 실행을 시작하지 않고 멈춥니다:")
            for problem in problems:
                print(f"  - {problem}")
            print(f"[sweep] 로그 파일: {log_dir / (row['run_id'] + '.log')}")
            sys.exit(1)
        append_row(csv_path, row)
        remaining = len(pending(grid)) + args.top * args.repeats
        hours = remaining * float(row["duration_s"]) / 3600
        print(f"[sweep] 점검 통과 (1회 {row['duration_s']}초, 가장 촘촘한 메시 기준).")
        print(f"[sweep] 이제 자리를 비워도 됩니다. 남은 실행 최대 {remaining}회 (1차 + 2차 반복), 최대 약 {hours:.1f}시간 예상.")

    def run_all(runs: list[dict], stage: str) -> None:
        todo = pending(runs)
        for i, params in enumerate(todo, start=1):
            print(f"[sweep] {stage} {i}/{len(todo)} {run_id_of(params)}", flush=True)
            row = run_one(params, log_dir)
            append_row(csv_path, row)
            print(f"[sweep]   -> {row['status']} ({row['duration_s']}초)", flush=True)

    run_all(grid, "1차")
    print_summary(csv_path)

    # Stage 2: repeat the best stage-1 combinations with the cloth shifted slightly
    stage1 = [r for r in load_done(csv_path).values() if r["seed"] == "0"]
    repeats = pick_repeats(stage1)
    if not repeats:
        print("\n[sweep] 1차에서 들어 올린 조합이 없어 2차 반복은 건너뜁니다.")
        return
    picked = sorted({combo_id(run_id_of(p)) for p in repeats})
    print(f"\n[sweep] 2차 반복 시작: {', '.join(picked)} 각 {args.repeats}번 (천 위치 ±{args.jitter * 1000:.0f} mm)")
    run_all(repeats, "2차")
    print_repeat_summary(csv_path)


if __name__ == "__main__":
    main()
