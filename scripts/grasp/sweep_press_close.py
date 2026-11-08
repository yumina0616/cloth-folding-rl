"""
Run press_close_grasp.py over a grid of settings (experiment 007).
Each combination runs in its own process. Results are appended to a CSV
right after each run, so the sweep can be stopped and resumed.

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

# Grid (lengths in m). Edit here to change the sweep.
OPEN_WIDTHS = [0.03, 0.05, 0.08]
PRESS_DEPTHS = [0.0, 0.003, 0.006]
THICKNESSES = [0.003, 0.006]
BEND_SCALES = [1.0, 0.1, 0.01]
EDGE_REFINEMENTS = [16, 32]

# The first run checks that the gripper opening and mesh settings are applied
PREFLIGHT = {"open_width": 0.03, "press_depth": 0.0, "thickness": 0.006, "bend_scale": 1.0, "edge_refinement": 32}

PARAM_KEYS = ["open_width", "press_depth", "thickness", "bend_scale", "edge_refinement"]
RESULT_KEYS = [
    "num_nodes", "inset", "open_gap", "left_in", "right_in", "left_z", "right_z",
    "press_gap", "ridge", "finger_gap", "hold_min_rise", "corner_rise", "passed",
]
CSV_FIELDS = ["run_id", "status", "duration_s", *PARAM_KEYS, *RESULT_KEYS]

parser = argparse.ArgumentParser(description="Grid sweep for the press-and-close grasp")
parser.add_argument("--out", type=Path, default=REPO / "logs" / "sweep_007", help="Folder for results.csv and run logs")
parser.add_argument("--timeout", type=float, default=900.0, help="Seconds before a stuck run is killed")
parser.add_argument("--retry_failed", action="store_true", help="Run ERROR/TIMEOUT combinations again")
parser.add_argument("--skip_preflight", action="store_true")
args = parser.parse_args()


def run_id_of(params: dict) -> str:
    return (
        f"w{params['open_width'] * 1000:.0f}_d{params['press_depth'] * 1000:.0f}"
        f"_t{params['thickness'] * 1000:.0f}_b{params['bend_scale']:g}_r{params['edge_refinement']}"
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
    cmd = [
        sys.executable, str(GRASP_SCRIPT),
        "--open_width", str(params["open_width"]),
        "--press_depth", str(params["press_depth"]),
        "--thickness", str(params["thickness"]),
        "--bend_scale", str(params["bend_scale"]),
        "--edge_refinement", str(params["edge_refinement"]),
        "--no_gui",
        "physics=physx",
    ]
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


def print_summary(csv_path: Path) -> None:
    rows = list(load_done(csv_path).values())
    ok = [r for r in rows if r["status"] == "OK"]
    print(f"\n[sweep] 완료: 총 {len(rows)}회, 정상 {len(ok)}회, 오류·시간초과 {len(rows) - len(ok)}회")
    # Rank: PASS first, then how high the cloth stayed, then the ridge height
    ok.sort(key=lambda r: (r["passed"] == "True", float(r["hold_min_rise"]), float(r["ridge"])), reverse=True)
    print("[sweep] 상위 10개 조합 (단위 mm, 손가락_천위=아니오면 손가락이 천 밖에 있었으므로 결과를 믿으면 안 됨)")
    for r in ok[:10]:
        on_cloth = float(r["left_in"]) > 0 and float(r["right_in"]) > 0
        print(
            f"  {r['run_id']:<28} {'성공' if r['passed'] == 'True' else '실패'}"
            f" | 유지높이={float(r['hold_min_rise']):.1f} | 봉우리={float(r['ridge']):.1f}"
            f" | 손가락간격={float(r['finger_gap']):.1f} | 누름간격={float(r['press_gap']):.1f}"
            f" | 손가락_천위={'예' if on_cloth else '아니오'}"
        )
    print(f"[sweep] 전체 결과 표: {csv_path}")


def main() -> None:
    log_dir = args.out / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"

    grid = [dict(zip(PARAM_KEYS, values)) for values in itertools.product(
        OPEN_WIDTHS, PRESS_DEPTHS, THICKNESSES, BEND_SCALES, EDGE_REFINEMENTS
    )]
    skip_status = {"OK", "NO_RESULT"} if args.retry_failed else {"OK", "NO_RESULT", "ERROR", "TIMEOUT"}

    def pending() -> list[dict]:
        done = load_done(csv_path)
        return [p for p in grid if done.get(run_id_of(p), {}).get("status") not in skip_status]

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
        remaining = len(pending())
        hours = remaining * float(row["duration_s"]) / 3600
        print(f"[sweep] 점검 통과 (1회 {row['duration_s']}초).")
        print(f"[sweep] 이제 자리를 비워도 됩니다. 남은 실행 {remaining}회, 약 {hours:.1f}시간 예상.")

    todo = pending()
    for i, params in enumerate(todo, start=1):
        print(f"[sweep] {i}/{len(todo)} {run_id_of(params)}", flush=True)
        row = run_one(params, log_dir)
        append_row(csv_path, row)
        print(f"[sweep]   -> {row['status']} ({row['duration_s']}초)", flush=True)

    print_summary(csv_path)


if __name__ == "__main__":
    main()
