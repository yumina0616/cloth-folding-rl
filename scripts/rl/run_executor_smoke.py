"""Smoke test for BatchedFoldExecutor: N environments fold the flat cloth like the demo's script.

Each environment does the scripted expert: grasp corner `source`, place it on the destination
corner's (x, y) as it was when the carry started. Prints per-environment grasp result, slip/drop,
placement miss, pair distances, success, and how long one pick-and-place took (= the cost of one
policy decision, the number PPO vs SAC depends on).

    ~/IsaacLab/isaaclab.sh -p scripts/rl/run_executor_smoke.py --num_envs 4 --viz none presets=isaacsim_physx
    ... --grasp_mode virtual                                     (check the executor without finger contact)
    ... --finger_friction 2.0 --finger_static_friction 3.0       (teleop #31 finger friction)

Cloth/table/finger settings default to configs/cloth_unified.json (same as the demo, --config none = off).
The placement differs from the demo's physical default (--place_mode mirror), which needs a destination
corner; here the hand target is "destination - live corner-to-hand offset" (the demo's --place_mode live).
Output lines start with SMOKE followed by JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401  (registers Isaac-Lift-Cloth-Franka)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "grasp"))

from rectangular_fold_geometry import rectangular_pairs  # noqa: E402
from run_config import DEFAULT_CONFIG, apply_config_defaults  # noqa: E402
from scripts.rl.fold_executor import BatchedFoldExecutor, configure_fold_cfg  # noqa: E402
from scripts.rl.fold_mdp import fold_success, pair_distances  # noqa: E402


def as_list(tensor, scale=1.0, digits=4):
    return [round(float(v) * scale, digits) for v in tensor.flatten()] if tensor.ndim <= 1 else \
        [as_list(row, scale, digits) for row in tensor]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--grasp_mode", choices=["physical", "virtual"], default="physical")
    parser.add_argument("--success_distance", type=float, default=0.02)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    # Cloth / table / finger (defaults from configs/cloth_unified.json, as in the demo)
    parser.add_argument("--cloth_x", type=float, default=0.50)
    parser.add_argument("--cloth_y", type=float, default=-0.05)
    parser.add_argument("--cloth_size", type=float, default=0.30)
    parser.add_argument("--mesh_refinement", type=float, default=32.0)
    parser.add_argument("--cloth_thickness", type=float, default=0.003)
    parser.add_argument("--cloth_density", type=float, default=1000.0 / 3)
    parser.add_argument("--youngs_modulus", type=float, default=2e5 / 3)
    parser.add_argument("--bend_stiffness", type=float, default=2e4 / 108)
    parser.add_argument("--poissons_ratio", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--cloth_friction", type=float, default=None)
    parser.add_argument("--cloth_static_friction", type=float, default=None)
    parser.add_argument("--table_friction", type=float, default=None)
    parser.add_argument("--table_static_friction", type=float, default=None)
    parser.add_argument("--table_length", type=float, default=1.5)
    parser.add_argument("--table_width", type=float, default=0.9)
    parser.add_argument("--open_width", type=float, default=0.04)
    parser.add_argument("--press_depth", type=float, default=0.0)
    parser.add_argument("--finger_friction", type=float, default=1.5)
    parser.add_argument("--finger_static_friction", type=float, default=2.0)
    parser.add_argument("--gripper_effort", type=float, default=70.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="JSON of cloth/table/finger defaults; 'none' = script defaults only")
    apply_config_defaults(parser, sys.argv[1:])
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if args.num_envs < 1:
        parser.error("--num_envs must be at least 1")
    for name in ("cloth", "table"):
        dynamic, static = getattr(args, f"{name}_friction"), getattr(args, f"{name}_static_friction")
        if static is not None and dynamic is None:
            parser.error(f"--{name}_static_friction needs --{name}_friction")
        if dynamic is not None:
            setattr(args, f"{name}_static_friction", dynamic if static is None else static)
    for name in ("cloth", "table", "finger"):
        dynamic, static = getattr(args, f"{name}_friction"), getattr(args, f"{name}_static_friction")
        if dynamic is not None and not (math.isfinite(dynamic) and math.isfinite(static) and 0 <= dynamic <= static):
            parser.error(f"{name} friction must satisfy 0 <= dynamic ({dynamic}) <= static ({static})")
    sys.argv = [sys.argv[0], *hydra_args]

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
        material.static_friction = args.cloth_static_friction
        material.dynamic_friction = args.cloth_friction
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    cfg.scene.deformable.init_state.pos = (args.cloth_x, args.cloth_y, 0.012)
    cfg.scene.table.spawn.size = (args.table_length, args.table_width, 1.05)
    if args.table_friction is not None:
        cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.table_static_friction, dynamic_friction=args.table_friction,
            restitution=0.0, friction_combine_mode="min")
    configure_fold_cfg(cfg, args.num_envs, grasp_mode=args.grasp_mode, open_width=args.open_width,
                       finger_friction=args.finger_friction, finger_static_friction=args.finger_static_friction,
                       gripper_effort=args.gripper_effort)
    table_z = float(cfg.scene.table.init_state.pos[2]) + 1.05 / 2  # table top (0 in this task)

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        env.reset()
        base = env.unwrapped
        marker = getattr(base.command_manager.get_term("deformable_pose"), "success_visualizer", None)
        if marker is not None:
            marker.set_visibility(False)
        with torch.inference_mode():
            executor = BatchedFoldExecutor(
                env, grasp_mode=args.grasp_mode, table_z=table_z, cloth_thickness=args.cloth_thickness,
                lift_height=args.lift_height, settle_seconds=args.settle_seconds, open_width=args.open_width,
                press_depth=args.press_depth, retries=args.retries)
            executor.hold(120)
            executor.detect_corners()
            corners = executor.corners()
            root = executor.robot.data.root_pos_w
            robot_xy = (root if isinstance(root, torch.Tensor) else root.torch)[:, :2]
            pairs = rectangular_pairs(corners[0], robot_xy[0])
            # Every environment is an identical copy, so the same pairs must come out everywhere
            for i in range(1, args.num_envs):
                if rectangular_pairs(corners[i], robot_xy[i]) != pairs:
                    raise RuntimeError(f"Environment {i} chose different corner pairs than environment 0")
            print("SMOKE " + json.dumps({"event": "start", "num_envs": args.num_envs, "grasp_mode": args.grasp_mode,
                                         "pairs": pairs, "distances_m": as_list(pair_distances(corners, pairs))}),
                  flush=True)

            still_ok = torch.ones(args.num_envs, dtype=torch.bool, device=base.device)
            for number, (source, destination) in enumerate(pairs, 1):
                start = time.perf_counter()
                ok, grasp_info = executor.grasp_corner(torch.full((args.num_envs,), source), active=still_ok)
                place_xy = executor.corners()[:, destination, :2].clone()  # scripted expert: onto the corner
                miss, place_info = executor.place_held_corner(place_xy, active=ok)
                elapsed = time.perf_counter() - start
                still_ok &= ok & place_info["placed"]
                print("SMOKE " + json.dumps({
                    "event": "placement", "number": number,
                    "grasp_ok": ok.tolist(), "attempts": grasp_info["attempts"].tolist(),
                    "rise_mm": as_list(grasp_info["rise_m"], 1000, 1),
                    "dropped": place_info["dropped"].tolist(),
                    "max_drift_mm": as_list(place_info["max_drift_m"], 1000, 1),
                    "miss_mm": as_list(miss, 1000, 1),
                    "distances_m": as_list(pair_distances(executor.corners(), pairs)),
                    "seconds": round(elapsed, 1),
                    "decisions_per_s": round(args.num_envs / elapsed, 3),
                }), flush=True)

            success = fold_success(executor.corners(), pairs, args.success_distance) & still_ok
            print("SMOKE " + json.dumps({"event": "done", "success": success.tolist(),
                                         "success_rate": float(success.float().mean())}), flush=True)
            executor.release_all()
        env.close()


if __name__ == "__main__":
    main()
