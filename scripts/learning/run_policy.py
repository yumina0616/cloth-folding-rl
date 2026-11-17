"""Run a BC/PPO checkpoint in Isaac, or continue it with PPO."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/grasp"))

import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

from teleop_cloth import ClothTeleop, configure
from scripts.learning.isaac_pick_place import IsaacPickPlace
from scripts.learning.pick_place_env import PickPlaceEnv
from scripts.learning.policy_contract import VERSION, normalize_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["evaluate", "ppo", "smoke"], default="evaluate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--grasp_mode", choices=["virtual", "physical"], default="virtual")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--n_steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open_width", type=float, default=.04)
    parser.add_argument("--grasp_radius", type=float)
    parser.add_argument("--success_distance", type=float, default=.02)
    parser.add_argument("--settle_seconds", type=float, default=1.)
    parser.add_argument("--mesh_refinement", type=float, default=32.)
    parser.add_argument("--lift_height", type=float, default=.12)
    parser.add_argument("--cloth_x", type=float, default=.50)
    parser.add_argument("--cloth_y", type=float, default=-.05)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    if args.grasp_radius is None:
        args.grasp_radius = .06 if args.grasp_mode == "physical" else .025
    if args.task != "smoke" and args.checkpoint is None:
        parser.error("--checkpoint is required for evaluation/PPO")
    if min(args.episodes, args.max_attempts, args.timesteps, args.n_steps, args.batch_size) < 1:
        parser.error("Counts must be positive")
    if args.n_steps < 2 or args.batch_size < 2 or args.n_steps % args.batch_size:
        parser.error("n_steps must be >=2 and divisible by batch_size >=2")
    if not 0 < args.open_width <= .08 or min(args.grasp_radius, args.success_distance,
            args.settle_seconds, args.mesh_refinement, args.lift_height) <= 0:
        parser.error("Invalid physical/control parameters")
    args.output_dir = (args.output_dir or ROOT / "runs/learning" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure(args)
    cfg.scene.deformable.init_state.pos = (args.cloth_x, args.cloth_y, .012)
    # configure uses teleoperation's non-resetting low-level environment.
    with launch_simulation(cfg, args):
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.buffers import RolloutBuffer
        low_env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        env = None
        try:
            with torch.no_grad():
                session = ClothTeleop(low_env, args)
            env = Monitor(PickPlaceEnv(IsaacPickPlace(session, args.lift_height), args.grasp_mode,
                                      args.max_attempts, args.success_distance))
            (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
            log_path = args.output_dir / "transitions.jsonl"

            def record(info, reward, terminated, truncated):
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**info, "reward": float(reward),
                                             "terminated": bool(terminated), "truncated": bool(truncated)}) + "\n")

            if args.task == "smoke":
                env.reset(seed=args.seed)
                source, destination = session.pairs[0]
                pick = session.corners()[source, :2].clone()
                if args.grasp_mode == "physical":
                    from physical_picker import as_tensor
                    pick = as_tensor(session.cloth.data.nodal_pos_w)[0, session.pinch_ids[source], :2].clone()
                place = session.corners()[destination, :2].clone()
                _, reward, terminated, truncated, info = env.step(normalize_action([*pick.cpu().tolist(), *place.cpu().tolist()]))
                record(info, reward, terminated, truncated)
                print(json.dumps({"smoke": "completed one real pick/place attempt", **info}), flush=True)
                return

            model = PPO.load(args.checkpoint, device=args.device)
            if getattr(model, "cloth_contract", None) != VERSION:
                raise ValueError("Checkpoint is not compatible with this cloth policy contract")
            if args.grasp_mode not in getattr(model, "cloth_training_modes", []):
                raise ValueError("Checkpoint has no training data for this grasp mode")
            model.set_env(env)
            if args.task == "ppo":
                # Rebuild rollout storage when overriding BC checkpoint settings.
                model.n_steps, model.batch_size = args.n_steps, args.batch_size
                model.rollout_buffer = RolloutBuffer(args.n_steps, model.observation_space, model.action_space,
                    device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
                class LogTransitions(BaseCallback):
                    def _on_step(self):
                        info = self.locals["infos"][0]
                        done = bool(self.locals["dones"][0])
                        truncated = bool(info.get("TimeLimit.truncated", False))
                        record(info, self.locals["rewards"][0], done and not truncated, truncated)
                        return True
                try:
                    model.learn(total_timesteps=args.timesteps, callback=LogTransitions())
                finally:
                    model.save(args.output_dir / "ppo_latest")
            else:
                successes = 0
                for episode in range(args.episodes):
                    obs, _ = env.reset(seed=args.seed + episode)
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        record({"evaluation_episode": episode, **info}, reward, terminated, truncated)
                        if terminated or truncated:
                            successes += int(info["is_success"])
                            print(f"episode={episode} success={info['is_success']} distances={info['pair_distances_m']}", flush=True)
                            break
                summary = {"episodes": args.episodes, "successes": successes, "success_rate": successes / args.episodes,
                           "grasp_mode": args.grasp_mode, "initial_xy": [args.cloth_x, args.cloth_y]}
                (args.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary), flush=True)
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                low_env.close()


if __name__ == "__main__":
    main()
