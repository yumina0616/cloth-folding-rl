"""Smoke test for BatchedFoldExecutor: N environments fold the flat cloth like the demo's script.

Each environment does the scripted expert: grasp corner `source`, place it on the destination
corner's (x, y). Prints per-environment pair distances, success, placement miss, and how long one
pick-and-place took (= the cost of one policy decision, the number PPO vs SAC depends on).

    ~/IsaacLab/isaaclab.sh -p scripts/rl/run_executor_smoke.py --num_envs 4 --viz none presets=isaacsim_physx

Virtual grasp, cloth settings = rectangular_fold_demo.py defaults on main (#19).
Output lines start with SMOKE followed by JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab_tasks  # noqa: F401  (registers Isaac-Lift-Cloth-Franka)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "grasp"))

from rectangular_fold_geometry import rectangular_pairs  # noqa: E402
from scripts.rl.fold_executor import BatchedFoldExecutor, configure_virtual_fold_cfg  # noqa: E402
from scripts.rl.fold_mdp import fold_success, pair_distances  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--success_distance", type=float, default=0.02)
    # Same defaults as rectangular_fold_demo.py on main
    parser.add_argument("--cloth_x", type=float, default=0.50)
    parser.add_argument("--cloth_y", type=float, default=-0.05)
    parser.add_argument("--cloth_size", type=float, default=0.30)
    parser.add_argument("--mesh_refinement", type=float, default=32.0)
    parser.add_argument("--cloth_thickness", type=float, default=0.003)
    parser.add_argument("--cloth_density", type=float, default=1000.0 / 3)
    parser.add_argument("--youngs_modulus", type=float, default=2e5 / 3)
    parser.add_argument("--bend_stiffness", type=float, default=2e4 / 108)
    parser.add_argument("--table_length", type=float, default=1.5)
    parser.add_argument("--table_width", type=float, default=0.9)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if args.num_envs < 1:
        parser.error("--num_envs must be at least 1")
    sys.argv = [sys.argv[0], *hydra_args]

    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    material = cfg.scene.deformable.spawn.physics_material
    material.surface_bend_stiffness = args.bend_stiffness
    material.youngs_modulus = args.youngs_modulus
    material.surface_thickness = args.cloth_thickness
    material.density = args.cloth_density
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    cfg.scene.deformable.init_state.pos = (args.cloth_x, args.cloth_y, 0.012)
    cfg.scene.table.spawn.size = (args.table_length, args.table_width, 1.05)
    configure_virtual_fold_cfg(cfg, args.num_envs)
    table_z = float(cfg.scene.table.init_state.pos[2]) + 1.05 / 2  # table top (0 in this task)

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        env.reset()
        base = env.unwrapped
        marker = getattr(base.command_manager.get_term("deformable_pose"), "success_visualizer", None)
        if marker is not None:
            marker.set_visibility(False)
        with torch.inference_mode():
            executor = BatchedFoldExecutor(env, table_z=table_z, cloth_thickness=args.cloth_thickness,
                                           lift_height=args.lift_height, settle_seconds=args.settle_seconds)
            executor.hold(120)
            executor.detect_corners()
            corners = executor.corners()
            robot_xy = executor.robot.data.root_pos_w
            robot_xy = (robot_xy if isinstance(robot_xy, torch.Tensor) else robot_xy.torch)[:, :2]
            pairs = rectangular_pairs(corners[0], robot_xy[0])
            # Every environment is an identical copy, so the same pairs must come out everywhere
            for i in range(1, args.num_envs):
                if rectangular_pairs(corners[i], robot_xy[i]) != pairs:
                    raise RuntimeError(f"Environment {i} chose different corner pairs than environment 0")
            print("SMOKE " + json.dumps({"event": "start", "num_envs": args.num_envs, "pairs": pairs,
                                         "distances_m": pair_distances(corners, pairs).tolist()}), flush=True)

            for number, (source, destination) in enumerate(pairs, 1):
                start = time.perf_counter()
                ok = executor.grasp_corner(torch.full((args.num_envs,), source))
                place_xy = executor.corners()[:, destination, :2].clone()  # scripted expert: onto the corner
                miss = executor.place_held_corner(place_xy, active=ok)
                elapsed = time.perf_counter() - start
                corners = executor.corners()
                print("SMOKE " + json.dumps({
                    "event": "placement", "number": number, "grasp_ok": ok.tolist(),
                    "miss_mm": (miss * 1000).round().tolist(),
                    "distances_m": pair_distances(corners, pairs).tolist(),
                    "seconds": round(elapsed, 1),
                    "decisions_per_s": round(args.num_envs / elapsed, 3),
                }), flush=True)

            success = fold_success(corners, pairs, args.success_distance)
            print("SMOKE " + json.dumps({"event": "done", "success": success.tolist(),
                                         "success_rate": float(success.float().mean())}), flush=True)
            executor.release_all()
        env.close()


if __name__ == "__main__":
    main()
