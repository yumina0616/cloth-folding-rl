"""Evaluate separate high/low BC checkpoints in the Isaac cloth-folding executor.

This runner replaces the legacy four-dimensional pick+place policy evaluation.
It runs ONE fresh Isaac episode per invocation because BatchedFoldExecutor and
HybridPPOBridge do not yet implement a safe full cloth/robot reset.

A limited --task ppo mode collects ONE complete on-policy episode and runs
one update for each separate actor-critic. It is a connection test, NOT a
validated multi-episode training loop: safe per-environment cloth reset and
long-horizon rollout batching remain unimplemented.

Example (Isaac Lab Python, from the repository root):
  ... -u scripts/learning/run_policy.py --task evaluate \
      --bc_dir runs/learning/bc_hierarchical --grasp_mode physical \
      --viz none presets=isaacsim_physx
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "scripts" / "grasp"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401: register Isaac-Lift-Cloth-Franka

from rectangular_fold_geometry import rectangular_pairs
from run_config import DEFAULT_CONFIG, apply_config_defaults
from scripts.learning.policy_contract import (
    VERSION, decode_action, decode_low_action,
    encode_observation, encode_low_observation,
    HIGH_OBS_SIZE, LOW_OBS_SIZE,
)
from scripts.rl.fold_executor import BatchedFoldExecutor, configure_fold_cfg
from scripts.rl.hybrid_ppo_bridge import HybridPPOBridge
from scripts.rl.fold_mdp import pair_distances


def _as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def load_bc_models(bc_dir, grasp_mode, device="cpu"):
    """Load two independently pretrained PPO actors and their low-action bounds."""
    from stable_baselines3 import PPO

    bc_dir = Path(bc_dir)
    bounds_file = bc_dir / "low_action_bounds.json"
    data = json.loads(bounds_file.read_text(encoding="utf-8"))
    if data.get("contract") != VERSION or data.get("action_frame") != "robot_root":
        raise ValueError("Low-level action bounds do not match the policy contract/coordinate frame")
    low = np.asarray(data["low"], dtype=np.float32)
    high = np.asarray(data["high"], dtype=np.float32)
    if low.shape != (8,) or high.shape != (8,) or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Invalid saved low-level action bounds")

    models = {}
    for level, expected_obs, expected_act in (
        ("high", HIGH_OBS_SIZE, 2), ("low", LOW_OBS_SIZE, 8)
    ):
        path = bc_dir / (level + "_bc_ppo.zip")
        model = PPO.load(path, device=device)
        if getattr(model, "cloth_contract", None) != VERSION:
            raise ValueError(f"{path}: model uses a different policy contract")
        if getattr(model, "cloth_policy_level", None) != level:
            raise ValueError(f"{path}: checkpoint belongs to a different policy level")
        if grasp_mode not in getattr(model, "cloth_training_modes", []):
            raise ValueError(f"{path}: no BC demonstrations for grasp_mode={grasp_mode}")
        if model.observation_space.shape != (expected_obs,) or model.action_space.shape != (expected_act,):
            raise ValueError(f"{path}: observation/action dimensions do not match")
        models[level] = model
    return models["high"], models["low"], low, high


def policy_observation(bridge, grasp_mode, level):
    """Encode the SAME observation order as the BC dataset (one environment only)."""
    state = bridge.observation()
    corners = _as_numpy(state["corners_w"])[0]
    if level == "high":
        return encode_observation(corners, bridge.pairs, grasp_mode)
    if level != "low":
        raise ValueError("Expected high or low policy")
    tcp = _as_numpy(state["tcp_w"])[0]
    previous = _as_numpy(state["ik_action_root"])[0]
    pair_idx = int(_as_numpy(state["pair_index"])[0])
    grasping = bool(_as_numpy(state["grasping"])[0])
    distances = _as_numpy(state["pair_distances_m"])[0]
    return encode_low_observation(corners, tcp, previous, pair_idx, grasping, distances)


def run_episode(bridge, high_model, low_model, grasp_mode, low_bounds, *, max_iterations):
    """Run one episode using frozen BC policies; no PPO gradient updates."""
    lower, upper = low_bounds
    log = []
    for _ in range(max_iterations):
        phase = int(bridge.phase[0].item())
        if phase == bridge.DONE:
            break
        if phase == bridge.PICK:
            obs = policy_observation(bridge, grasp_mode, "high")
            output, _ = high_model.predict(obs, deterministic=True)
            pick_xy = decode_action(output)
            _, info = bridge.begin_pick(torch.as_tensor(pick_xy[None, :], device=bridge.device))
            log.append({"event": "pick", "pair_index": int(bridge.pair_index[0]),
                        "world_pick_xy": pick_xy.tolist(),
                        "initialized": bool(info["pick_initialized"][0])})
        else:
            # SETTLE advances the simulator without taking an additional actor decision.
            if phase == bridge.SETTLE:
                root_action = bridge.ex.action.detach().clone()
            else:
                obs = policy_observation(bridge, grasp_mode, "low")
                output, _ = low_model.predict(obs, deterministic=True)
                command = decode_low_action(output, lower, upper)
                root_action = torch.as_tensor(command[None, :], device=bridge.device)
            _, reward, _, info = bridge.step_low_level(root_action)
            if bool(info["released"][0]) or bool(info["fold_success"][0]) or bool(info["invalid"][0]):
                log.append({"event": "low_transition", "phase": int(info["phase"][0]),
                            "pair_index": int(info["pair_index"][0]),
                            "released": bool(info["released"][0]),
                            "invalid": bool(info["invalid"][0]),
                            "reward": float(reward[0])})
    else:
        raise RuntimeError("Evaluation exceeded max_iterations; episode did not finish")
    corners = bridge.ex.corners()
    final_distances = pair_distances(corners, bridge.pairs)[0]
    return {
        "completed": int(bridge.phase[0]) == bridge.DONE,
        "invalid": bool(bridge.invalid[0]),
        "pair_distances_m": _as_numpy(final_distances).tolist(),
        "success": bool((final_distances <= bridge.threshold).all() and not bool(bridge.invalid[0])
                        and int(bridge.pair_index[0]) == 1),
        "transitions": log,
    }




def collect_ppo_episode(bridge, high_model, low_model, grasp_mode, low_bounds, *, max_iterations):
    """Collect one fresh, complete on-policy episode from the existing bridge.

    High-level decisions are temporal options: their reward is the sum of the
    subsequent low-level and settling rewards. A low-level trajectory ends on
    each release/failure, even if the full fold proceeds to another pick.
    Never include settling frames as new actor decisions.
    """
    from scripts.rl.ppo_policy_update import PPOTransition, sample_action

    lower, upper = low_bounds
    high_transitions = []
    low_episodes = []
    current_low = []
    pending_high = None
    for _ in range(max_iterations):
        phase = int(bridge.phase[0].item())
        if phase == bridge.DONE:
            break
        if phase == bridge.PICK:
            if pending_high is not None or current_low:
                raise RuntimeError("Previous option has not finished")
            obs = policy_observation(bridge, grasp_mode, "high")
            raw, value, log_prob = sample_action(high_model, obs)
            pick_xy = decode_action(raw)
            _, info = bridge.begin_pick(torch.as_tensor(pick_xy[None, :], device=bridge.device))
            pending_high = PPOTransition(obs, raw, log_prob, value, 0.0, False)
            if not bool(info["pick_initialized"][0]):
                # Failed scripted pick: no low-level action exists for this option.
                pending_high.reward -= 1.0
                pending_high.done = True
                high_transitions.append(pending_high)
                pending_high = None
                break
            continue

        if phase == bridge.CARRY:
            obs = policy_observation(bridge, grasp_mode, "low")
            raw, value, log_prob = sample_action(low_model, obs)
            command = decode_low_action(raw, lower, upper)
            root_action = torch.as_tensor(command[None, :], device=bridge.device)
            _, rewards, _, info = bridge.step_low_level(root_action)
            reward = float(rewards[0].item())
            current_low.append(PPOTransition(obs, raw, log_prob, value, reward, False))
        elif phase == bridge.SETTLE:
            # No new policy action: give delayed settling reward to the most
            # recent real action (and to its high-level pick option).
            if not current_low:
                raise RuntimeError("Settling without any low-level decision")
            root_action = bridge.ex.action.detach().clone()
            _, rewards, _, info = bridge.step_low_level(root_action)
            reward = float(rewards[0].item())
            current_low[-1].reward += reward
        else:
            raise RuntimeError(f"Unknown bridge phase: {phase}")

        pending_high.reward += reward
        new_phase = int(bridge.phase[0].item())
        if new_phase in (bridge.PICK, bridge.DONE):
            if not current_low:
                raise RuntimeError("Completed option without low-level actions")
            current_low[-1].done = True
            low_episodes.append(current_low)
            current_low = []
            pending_high.done = new_phase == bridge.DONE
            high_transitions.append(pending_high)
            pending_high = None
    else:
        raise RuntimeError("PPO rollout exceeded max_iterations; discarded incomplete data")

    if int(bridge.phase[0].item()) != bridge.DONE or pending_high is not None or current_low:
        raise RuntimeError("PPO rollout incomplete; do not update on partial trajectories")
    if not high_transitions or not high_transitions[-1].done:
        raise RuntimeError("Missing terminal high-level transition")
    # Failed first picks can train high alone, but the two-policy updater
    # explicitly requires both policies to have at least one real sample.
    if not low_episodes:
        raise RuntimeError("No low-level samples; skip the two-policy update")
    return [high_transitions], low_episodes


def train_one_ppo_episode(bridge, high_model, low_model, grasp_mode, low_bounds,
                          *, max_iterations, output_dir, epochs=4, minibatch_size=64):
    """Collect then update both BC-initialized PPO policies ONCE.

    Only one Isaac scene/episode is supported per process, because the current
    executor has no verified cloth+robot reset. Call a fresh process for another
    episode; do not treat this as a full multi-episode trainer.
    """
    from scripts.rl.ppo_policy_update import update_two_policies

    high_episodes, low_episodes = collect_ppo_episode(
        bridge, high_model, low_model, grasp_mode, low_bounds,
        max_iterations=max_iterations,
    )
    stats = update_two_policies(high_model, low_model, high_episodes,
                                low_episodes, epochs=epochs,
                                minibatch_size=minibatch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    high_model.save(output_dir / "high_ppo_latest")
    low_model.save(output_dir / "low_ppo_latest")
    # Save decoding bounds alongside the updated models for later evaluation.
    from scripts.learning.policy_contract import VERSION
    lower, upper = low_bounds
    (output_dir / "low_action_bounds.json").write_text(
        json.dumps({"contract": VERSION, "action_frame": "robot_root",
                    "action_order": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
                    "low": np.asarray(lower).tolist(),
                    "high": np.asarray(upper).tolist()}, indent=2), encoding="utf-8")
    (output_dir / "ppo_update_report.json").write_text(
        json.dumps({"warning": "one episode only; not a multi-episode PPO training run",
                    "high_options": sum(len(ep) for ep in high_episodes),
                    "low_decisions": sum(len(ep) for ep in low_episodes),
                    "update": stats}, indent=2), encoding="utf-8")
    return stats


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["evaluate", "ppo"], default="evaluate")
    parser.add_argument("--ppo_output", type=Path, default=Path("runs/learning/ppo_single_episode"))
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_minibatch_size", type=int, default=64)
    parser.add_argument("--bc_dir", type=Path, default=Path("runs/learning/bc_hierarchical"))
    parser.add_argument("--grasp_mode", choices=["virtual", "physical"], default="physical")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_low_steps", type=int, default=1000)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--settle_seconds", type=float, default=1.0)
    parser.add_argument("--success_distance", type=float, default=.02)
    parser.add_argument("--cloth_x", type=float, default=.5)
    parser.add_argument("--cloth_y", type=float, default=-.05)
    parser.add_argument("--cloth_size", type=float, default=.30)
    parser.add_argument("--mesh_refinement", type=float, default=32.)
    parser.add_argument("--cloth_thickness", type=float, default=.003)
    parser.add_argument("--cloth_density", type=float, default=1000. / 3.)
    parser.add_argument("--youngs_modulus", type=float, default=2e5 / 3.)
    parser.add_argument("--bend_stiffness", type=float, default=2e4 / 108.)
    parser.add_argument("--poissons_ratio", type=float)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--cloth_friction", type=float)
    parser.add_argument("--cloth_static_friction", type=float)
    parser.add_argument("--table_friction", type=float)
    parser.add_argument("--table_static_friction", type=float)
    parser.add_argument("--table_length", type=float, default=1.5)
    parser.add_argument("--table_width", type=float, default=.9)
    parser.add_argument("--open_width", type=float, default=.04)
    parser.add_argument("--press_depth", type=float, default=0.)
    parser.add_argument("--finger_friction", type=float, default=1.5)
    parser.add_argument("--finger_static_friction", type=float, default=2.)
    parser.add_argument("--gripper_effort", type=float, default=70.)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main():
    parser = build_parser()
    apply_config_defaults(parser, sys.argv[1:])
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if args.num_envs != 1 or args.episodes != 1:
        parser.error("Current evaluation supports exactly one environment and one episode per process")
    if args.ppo_epochs < 1 or args.ppo_minibatch_size < 1:
        parser.error("PPO epochs and minibatch size must be positive")
    if args.max_low_steps < 1 or args.max_iterations < 1 or args.settle_seconds < 0:
        parser.error("Invalid step limits / settling duration")
    if args.success_distance <= 0:
        parser.error("success_distance must be positive")
    sys.argv = [sys.argv[0], *hydra_args]
    high_model, low_model, lower, upper = load_bc_models(args.bc_dir, args.grasp_mode, args.device)

    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    material = cfg.scene.deformable.spawn.physics_material
    material.surface_bend_stiffness = args.bend_stiffness
    material.youngs_modulus = args.youngs_modulus
    material.surface_thickness = args.cloth_thickness
    material.density = args.cloth_density
    if args.poissons_ratio is not None:
        material.poissons_ratio = args.poissons_ratio
    if args.damping is not None:
        material.elasticity_damping = args.damping
        material.bend_damping = args.damping
    if args.cloth_friction is not None:
        material.dynamic_friction = args.cloth_friction
        material.static_friction = (args.cloth_static_friction if args.cloth_static_friction is not None else args.cloth_friction)
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    cfg.scene.deformable.init_state.pos = (args.cloth_x, args.cloth_y, .012)
    cfg.scene.table.spawn.size = (args.table_length, args.table_width, 1.05)
    if args.table_friction is not None:
        cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=(args.table_static_friction if args.table_static_friction is not None else args.table_friction),
            dynamic_friction=args.table_friction, restitution=0., friction_combine_mode="min")
    configure_fold_cfg(cfg, 1, grasp_mode=args.grasp_mode, open_width=args.open_width,
                       finger_friction=args.finger_friction, finger_static_friction=args.finger_static_friction,
                       gripper_effort=args.gripper_effort)
    table_top = float(cfg.scene.table.init_state.pos[2]) + 1.05 / 2

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        try:
            env.reset()
            with torch.no_grad():
                executor = BatchedFoldExecutor(
                    env, grasp_mode=args.grasp_mode, table_z=table_top,
                    cloth_thickness=args.cloth_thickness, open_width=args.open_width,
                    press_depth=args.press_depth, retries=args.retries)
                executor.hold(120)
                executor.detect_corners()
                initial_corners = executor.corners()
                root = executor.robot.data.root_pos_w
                root_xy = _as_numpy(root)[0, :2]
                pairs = rectangular_pairs(initial_corners[0],
                                          torch.as_tensor(root_xy, device=executor.device))
                bridge = HybridPPOBridge(executor, pairs,
                                         success_threshold=args.success_distance,
                                         settle_seconds=args.settle_seconds,
                                         max_low_steps=args.max_low_steps)
                if args.task == "ppo":
                    stats = train_one_ppo_episode(
                        bridge, high_model, low_model, args.grasp_mode,
                        (lower, upper), max_iterations=args.max_iterations,
                        output_dir=args.ppo_output, epochs=args.ppo_epochs,
                        minibatch_size=args.ppo_minibatch_size)
                    print("PPO_SINGLE_EPISODE " + json.dumps(stats, ensure_ascii=False), flush=True)
                else:
                    result = run_episode(bridge, high_model, low_model,
                                         args.grasp_mode, (lower, upper),
                                         max_iterations=args.max_iterations)
                    print("BC_EVAL " + json.dumps(result, ensure_ascii=False), flush=True)
                executor.release_all()
        finally:
            env.close()


if __name__ == "__main__":
    main()
