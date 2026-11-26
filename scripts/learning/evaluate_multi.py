"""Compare frozen BC and PPO policies on matched seeds."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def evaluate(args, current, seed, log):
    cmd = [
        str(args.isaaclab),
        "-p",
        "scripts/learning/run_policy.py",
        "--task", "evaluate",
        "--bc_dir", str(current),
        "--seed", str(seed),
        "--pos_jitter", str(args.pos_jitter),
        "--yaw_jitter", str(args.yaw_jitter),
        "--grasp_mode", "physical",
        "--num_envs", "1",
        "--episodes", "1",
        "--max_low_steps", str(args.max_low_steps),
        "--max_iterations", str(args.max_iterations),
        "--success_distance", "0.05",
        "--shape_iou_threshold", "0.75",
        "--cloth_x", "0.5",
        "--cloth_y", "-0.05",
        "--cloth_size", "0.3",
        "--cloth_thickness", "0.003",
        "--cloth_density", "180",
        "--mesh_refinement", "32",
        "--youngs_modulus", "1000000",
        "--bend_stiffness", "185.1851851852",
        "--poissons_ratio", "0.25",
        "--damping", "0.25",
        "--cloth_friction", "10.0",
        "--cloth_static_friction", "10.5",
        "--table_friction", "0.6",
        "--table_static_friction", "0.65",
        "--table_length", "1.5",
        "--table_width", "0.9",
        "--finger_friction", "1.5",
        "--finger_static_friction", "2.0",
        "--gripper_effort", "70",
        "--open_width", "0.04",
        "--press_depth", "0.0",
        "--retries", "1",
        "--settle_seconds", "0.5",
        "--viz", "none",
        "presets=isaacsim_physx",
    ]

    with log.open("w") as f:
        subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    lines = log.read_text(errors="replace").splitlines()
    results = [
        json.loads(line[len("BC_EVAL "):])
        for line in lines
        if line.startswith("BC_EVAL ")
    ]

    if len(results) != 1:
        raise RuntimeError(f"Missing or duplicate BC_EVAL: {log}")

    result = results[0]

    if not result["completed"]:
        raise RuntimeError(f"Incomplete evaluation: {log}")

    initial = [
        json.loads(line[len("[INIT] "):])
        for line in lines
        if line.startswith("[INIT] {")
    ]

    if len(initial) != 1 or initial[0]["seed"] != seed:
        raise RuntimeError(f"Invalid initial condition: {log}")

    result["initial"] = initial[0]
    result["seed"] = seed

    events = result["transitions"]

    for key in ("slipped", "dropped", "timeout", "lift_verified"):
        result[key] = any(e.get(key, False) for e in events)

    result["pick_failed"] = any(
        e["event"] == "pick" and not e["initialized"]
        for e in events
    )

    return result


def summarize(results):
    n = len(results)

    return {
        "episodes": n,
        "success_count": sum(r["success"] for r in results),
        "success_rate": sum(r["success"] for r in results) / n,
        "mean_iou": statistics.mean(
            r["shape_metrics"]["iou"] for r in results
        ),
        "mean_corner_distance_m": statistics.mean(
            statistics.mean(r["pair_distances_m"])
            for r in results
        ),
        "slip_rate": sum(r["slipped"] for r in results) / n,
        "drop_rate": sum(r["dropped"] for r in results) / n,
        "timeout_rate": sum(r["timeout"] for r in results) / n,
        "pick_failure_rate": sum(
            r["pick_failed"] for r in results
        ) / n,
        "lift_rate": sum(r["lift_verified"] for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bc_dir", type=Path, required=True)
    parser.add_argument("--ppo_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--max_low_steps", type=int, default=5000)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--pos_jitter", type=float, default=0.02)
    parser.add_argument("--yaw_jitter", type=float, default=15.0)
    parser.add_argument(
        "--isaaclab", type=Path,
        default=Path.home() / "IsaacLab/isaaclab.sh"
    )

    args = parser.parse_args()

    if min(args.episodes, args.max_low_steps, args.max_iterations) < 1:
        parser.error("Counts and limits must be positive")

    models = {
        "bc": args.bc_dir.resolve(),
        "ppo": args.ppo_dir.resolve(),
    }

    for checkpoint in models.values():
        for name in (
            "high_bc_ppo.zip",
            "low_bc_ppo.zip",
            "low_action_bounds.json",
        ):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    results = {"bc": [], "ppo": []}

    for name in models:
        (output / name).mkdir()

    for seed in range(args.start_seed,
                      args.start_seed + args.episodes):

        for name, current in models.items():
            folder = output / name

            print(
                f"[EVALUATE] model={name} seed={seed}",
                flush=True
            )

            result = evaluate(
                args, current, seed,
                folder / f"episode_{seed}.log"
            )

            (folder / f"episode_{seed}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )

            results[name].append(result)

            print("[RESULT] " + json.dumps({
                "model": name,
                "seed": seed,
                "success": result["success"],
                "iou": result["shape_metrics"]["iou"],
                "slipped": result["slipped"],
                "timeout": result["timeout"],
            }), flush=True)

        if results["bc"][-1]["initial"] != results["ppo"][-1]["initial"]:
            raise RuntimeError(f"Unmatched initial conditions: {seed}")

    summary = {
        "bc": summarize(results["bc"]),
        "ppo": summarize(results["ppo"]),
        "models": {k: str(v) for k, v in models.items()},
        "seeds": list(range(
            args.start_seed,
            args.start_seed + args.episodes
        )),
    }

    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )

    print("[COMPARISON] " + json.dumps(summary), flush=True)
    print("[COMPLETE]", output, flush=True)


if __name__ == "__main__":
    main()
