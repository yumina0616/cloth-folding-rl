"""Multi-episode PPO training using fresh Isaac processes."""
import argparse
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

    args = parser.parse_args()

    if min(args.rounds, args.episodes_per_round,
           args.ppo_epochs, args.minibatch_size,
           args.max_low_steps, args.max_iterations) < 1:
        parser.error("Counts and step limits must be positive")

    if not 0 <= args.yaw_jitter < 45 or args.pos_jitter < 0:
        parser.error("Invalid initial pose jitter")

    current = args.bc_dir.resolve()
    output = args.output.resolve()

    # Prevent mixing old and new experiment data.
    output.mkdir(parents=True, exist_ok=False)

    for name in (
        "high_bc_ppo.zip",
        "low_bc_ppo.zip",
        "low_action_bounds.json",
    ):
        if not (current / name).is_file():
            raise FileNotFoundError(current / name)

    for r in range(args.rounds):
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

        (round_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )

        # The updated policy becomes the next round's policy.
        current = checkpoint

        print(
            "[PPO_ROUND] " + json.dumps(report),
            flush=True
        )

    shutil.copytree(current, output / "final")

    print(
        "[COMPLETE] Final model:",
        output / "final",
        flush=True
    )


if __name__ == "__main__":
    main()
