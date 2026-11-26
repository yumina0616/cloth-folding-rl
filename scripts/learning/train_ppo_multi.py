"""Multi-episode PPO training using fresh Isaac processes."""
import argparse
import time
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.rl.ppo_policy_update import (
    PPOTransition,
    ppo_update,
)
from scripts.learning.policy_contract import VERSION

ROOT = Path(__file__).resolve().parents[2]


def deserialize(episodes):
    return [
        [
            PPOTransition(
                observation=np.asarray(
                    t["observation"], dtype=np.float32),
                action=np.asarray(
                    t["action"], dtype=np.float32),
                log_prob=float(t["log_prob"]),
                value=float(t["value"]),
                reward=float(t["reward"]),
                done=bool(t["done"]),
            )
            for t in ep
        ]
        for ep in episodes
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bc_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes_per_round", type=int, default=3)
    parser.add_argument("--start_seed", type=int, default=100)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--max_low_steps", type=int, default=5000)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--pos_jitter", type=float, default=0.02)
    parser.add_argument("--yaw_jitter", type=float, default=15.0)
    parser.add_argument(
        "--isaaclab", type=Path,
        default=Path.home() / "IsaacLab/isaaclab.sh"
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last completed round"
    )

    args = parser.parse_args()

    if min(args.rounds, args.episodes_per_round,
           args.ppo_epochs, args.minibatch_size,
           args.max_low_steps, args.max_iterations) < 1:
        parser.error("Counts and step limits must be positive")

    if not 0 <= args.yaw_jitter < 45 or args.pos_jitter < 0:
        parser.error("Invalid initial pose jitter")

    current = args.bc_dir.resolve()
    output = args.output.resolve()

    settings = {
        "bc_dir": str(current),
        "episodes_per_round": args.episodes_per_round,
        "start_seed": args.start_seed,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "max_low_steps": args.max_low_steps,
        "max_iterations": args.max_iterations,
        "pos_jitter": args.pos_jitter,
        "yaw_jitter": args.yaw_jitter,
        "isaaclab": str(args.isaaclab.resolve()),
    }

    start_round = 0
    settings_path = output / "training_config.json"

    if args.resume:
        if not settings_path.is_file():
            raise RuntimeError(
                "Resume requires training_config.json "
                "from a compatible training run"
            )

        previous = json.loads(settings_path.read_text())

        if previous != settings:
            raise ValueError(
                "Training settings differ from saved settings"
            )

        round_dirs = sorted(
            d for d in output.glob("round_*")
            if d.is_dir() and d.name[6:].isdigit()
        )

        incomplete = []

        for folder in round_dirs:
            number = int(folder.name[6:])
            report_path = folder / "report.json"
            checkpoint = folder / "checkpoint"

            complete = (
                report_path.is_file()
                and all(
                    (checkpoint / name).is_file()
                    for name in (
                        "high_bc_ppo.zip",
                        "low_bc_ppo.zip",
                        "low_action_bounds.json",
                    )
                )
            )

            if not complete:
                incomplete.append(folder)
                continue

            report = json.loads(report_path.read_text())

            if number != start_round + 1 or incomplete:
                raise RuntimeError(
                    "Checkpoint sequence is not contiguous"
                )

            if (
                report["round"] != number
                or len(report["episodes"])
                != args.episodes_per_round
            ):
                raise RuntimeError("Invalid round report")

            start_round = number
            current = checkpoint

        if args.rounds < start_round:
            raise ValueError(
                "Requested total rounds is below completed rounds"
            )

        for folder in incomplete:
            backup = output / (
                f"interrupted_{folder.name}_{time.time_ns()}"
            )
            folder.rename(backup)
            print(
                "[RECOVERY] Preserved incomplete round:",
                backup, flush=True
            )

        print(
            f"[RESUME] Completed rounds: {start_round}",
            flush=True
        )
        print(
            f"[RESUME] Next round: {start_round + 1}",
            flush=True
        )
        print(
            f"[RESUME] Checkpoint: {current}",
            flush=True
        )

    else:
        output.mkdir(parents=True, exist_ok=False)
        tmp = settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings, indent=2))
        tmp.replace(settings_path)

    for name in (
        "high_bc_ppo.zip",
        "low_bc_ppo.zip",
        "low_action_bounds.json",
    ):
        if not (current / name).is_file():
            raise FileNotFoundError(current / name)

    for r in range(start_round, args.rounds):
        round_dir = output / f"round_{r+1:04d}"
        round_dir.mkdir()

        rollout_dir = round_dir / "rollouts"
        rollout_dir.mkdir()

        high_episodes = []
        low_episodes = []
        episode_reports = []

        # Keep the policy fixed throughout this rollout batch.
        for e in range(args.episodes_per_round):
            seed = (
                args.start_seed
                + r * args.episodes_per_round
                + e
            )

            rollout = rollout_dir / f"episode_{seed}.json"
            log = rollout_dir / f"episode_{seed}.log"

            cmd = [
                str(args.isaaclab),
                "-p",
                "scripts/learning/run_policy.py",
                "--task", "collect",
                "--bc_dir", str(current),
                "--rollout_output", str(rollout),
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

            print(
                f"[COLLECT] round={r+1} "
                f"episode={e+1} seed={seed}",
                flush=True
            )

            with log.open("w") as f:
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

            data = json.loads(rollout.read_text())

            if data["contract"] != VERSION:
                raise ValueError("Policy contract mismatch")
            if data["seed"] != seed:
                raise ValueError("Rollout seed mismatch")

            high = deserialize(data["high"])
            low = deserialize(data["low"])

            high_episodes.extend(high)
            low_episodes.extend(low)

            report = {
                "seed": seed,
                "metrics": data["metrics"],
                "x": data["start_x"],
                "y": data["start_y"],
                "yaw_deg": data["measured_yaw_deg"],
                "high_samples": sum(map(len, high)),
                "low_samples": sum(map(len, low)),
                "reward": sum(
                    t.reward for ep in high for t in ep
                ),
            }

            episode_reports.append(report)

            print(
                "[ROLLOUT] " + json.dumps(report),
                flush=True
            )

        # Update only after collecting all fresh episodes.
        torch.manual_seed(args.start_seed + r)
        np.random.seed(args.start_seed + r)

        high_model = PPO.load(
            current / "high_bc_ppo.zip", device="cpu")
        low_model = PPO.load(
            current / "low_bc_ppo.zip", device="cpu")

        update_args = {
            "epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
        }

        high_stats = ppo_update(
            high_model, high_episodes, **update_args)

        if low_episodes:
            low_stats = ppo_update(
                low_model, low_episodes, **update_args)
        else:
            low_stats = {
                "samples": 0,
                "updates": 0,
                "skipped": True,
            }

        checkpoint = round_dir / "checkpoint"
        checkpoint.mkdir()

        high_model.save(checkpoint / "high_bc_ppo")
        low_model.save(checkpoint / "low_bc_ppo")

        shutil.copy2(
            current / "low_action_bounds.json",
            checkpoint / "low_action_bounds.json",
        )

        report = {
            "round": r + 1,
            "input_checkpoint": str(current),
            "episodes": episode_reports,
            "high_update": high_stats,
            "low_update": low_stats,
        }

        report_path = round_dir / "report.json"
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        tmp.replace(report_path)

        print(
            "[CHECKPOINT] Saved:", checkpoint,
            flush=True
        )

        # The updated policy becomes the next round's policy.
        current = checkpoint

        print(
            "[PPO_ROUND] " + json.dumps(report),
            flush=True
        )

    final_dir = output / "final"
    temp_final = output / f"final_tmp_{time.time_ns()}"

    shutil.copytree(current, temp_final)

    if final_dir.exists():
        backup = output / f"previous_final_{time.time_ns()}"
        final_dir.rename(backup)

    temp_final.rename(final_dir)

    print(
        "[COMPLETE] Final model:",
        output / "final",
        flush=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\\n[INTERRUPTED] Training stopped by user. "
            "Completed round checkpoints are preserved. "
            "Use --resume to continue.",
            flush=True
        )
        sys.exit(130)
