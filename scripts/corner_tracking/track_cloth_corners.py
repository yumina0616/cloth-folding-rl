"""
Track and visualize the four corners of the cloth.

The corner indices are detected once after reset,
then the same cloth vertices are tracked during simulation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401 -- registers the task with Gymnasium

from cloth_corner_tracker import (
    CORNER_LABELS,
    CornerMotionVerifier,
    detect_corner_vertex_indices,
    get_corner_world_positions,
)


TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Detect and track the four cloth corner nodes")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--print_interval", type=int, default=30)
parser.add_argument("--movement_threshold", type=float, default=0.0005, help="Per-corner pass threshold [m]")
parser.add_argument("--csv", type=str, default="", help="Optional CSV output path")
parser.add_argument("--cloth_yaw", type=float, default=0.0, help="Initial cloth yaw angle [degrees]")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def configure_scene():
    """Create a flat towel scene with all four corners supported by the table."""
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0
    env_cfg.scene.support_neg_y = None
    env_cfg.scene.support_pos_y = None
    # The default Newton coupler still references the two supports above.
    # Remove those body patterns as well so the scene can initialize without
    # bodies that were intentionally disabled for this tracking test.
    physics_cfg = getattr(env_cfg.sim.physics, "default", env_cfg.sim.physics)
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    if solver_cfg is not None:
        for entry in getattr(solver_cfg, "entries", ()):
            entry.bodies = [body for body in entry.bodies if "Support" not in body]
        for proxy in getattr(solver_cfg, "proxies", ()):
            proxy.bodies = [body for body in proxy.bodies if "Support" not in body]
    env_cfg.scene.table.spawn.size = (1.3, 0.5, 1.05)
    env_cfg.scene.table.spawn.visible = True
    # Place the towel at the center of the table
    env_cfg.scene.deformable.init_state.pos = (0.5, 0.0, 0.012)
    half_yaw = math.radians(args_cli.cloth_yaw) * 0.5
    # Isaac Lab 3.0 uses (x, y, z, w) quaternions.
    env_cfg.scene.deformable.init_state.rot = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    towel_material = env_cfg.scene.deformable.spawn.physics_material
    towel_material.surface_thickness = 0.004
    towel_material.density = 150.0
    towel_material.static_friction = 1.0
    towel_material.dynamic_friction = 0.8

    env_cfg.commands.deformable_pose.debug_vis = False
    env_cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    env_cfg.events.reset_deformable = None
    env_cfg.curriculum.gravity = None
    env_cfg.curriculum.action_rate = None
    env_cfg.actions = type(env_cfg)().actions.ik
    for term_name in vars(env_cfg.terminations):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)

    env_cfg.viewer.eye = (1.2, 0.65, 0.55)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.03)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)
    return env_cfg


def create_corner_markers() -> VisualizationMarkers:
    colors = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.4, 1.0), (1.0, 0.8, 0.1))
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ClothCorners",
        markers={
            label: sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for label, color in zip(CORNER_LABELS, colors, strict=True)
        },
    )
    return VisualizationMarkers(cfg)


def make_hold_action(env) -> torch.Tensor:
    """Keep the robot at its current pose while gravity moves the towel."""
    action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    ee_frame = env.unwrapped.scene["ee_frame"]
    action[:, :3] = ee_frame.data.target_pos_w.torch[..., 0, :] - env.unwrapped.scene.env_origins
    action[:, 3:7] = ee_frame.data.target_quat_w.torch[..., 0, :]
    action[:, 7] = 1.0
    return action


def format_positions(positions: torch.Tensor) -> str:
    values = positions.detach().cpu().tolist()
    return " | ".join(
        f"{label}=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f})"
        for label, xyz in zip(CORNER_LABELS, values, strict=True)
    )


def main():
    if args_cli.num_steps <= 0:
        raise ValueError("--num_steps must be positive")
    if args_cli.print_interval <= 0:
        raise ValueError("--print_interval must be positive")
    if args_cli.movement_threshold < 0.0:
        raise ValueError("--movement_threshold cannot be negative")
    if not math.isfinite(args_cli.cloth_yaw):
        raise ValueError("--cloth_yaw must be finite")

    env_cfg = configure_scene()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.unwrapped.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        env.reset()

        cloth = env.unwrapped.scene["deformable"]
        env_origin = env.unwrapped.scene.env_origins[0]
        
        # Detect corner vertices once after reset
        initial_nodes_w = cloth.data.nodal_pos_w.torch[0]
        corner_indices = detect_corner_vertex_indices(initial_nodes_w)

        # Track the same vertex indices throughout the simulation
        initial_positions_w = get_corner_world_positions(initial_nodes_w, corner_indices).clone()
        # Verify actual simulated nodes against the requested flat, rotated mesh.
        # Movement alone can pass even when the initial rotation is incorrect.
        width, height = env_cfg.scene.deformable.spawn.size
        expected_xy = initial_positions_w.new_tensor(
            ((-width / 2, -height / 2), (-width / 2, height / 2),
             (width / 2, height / 2), (width / 2, -height / 2))
        )
        yaw = math.radians(args_cli.cloth_yaw)
        rotation = initial_positions_w.new_tensor(
            ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
        )
        expected_positions = initial_positions_w.new_zeros((4, 3))
        expected_positions[:, :2] = expected_xy @ rotation.T
        expected_positions += initial_positions_w.new_tensor(env_cfg.scene.deformable.init_state.pos) + env_origin
        nearest_errors, nearest_indices = torch.cdist(initial_positions_w, expected_positions).min(dim=1)
        max_error = float(nearest_errors.max().item())
        if max_error > 1.0e-5 or torch.unique(nearest_indices).numel() != 4:
            raise RuntimeError(
                f"Initial corners do not match the requested flat mesh at yaw={args_cli.cloth_yaw:g} "
                f"degrees (maximum error={max_error:.6f} m)"
            )
        print(f"[check] PASS: yaw={args_cli.cloth_yaw:g} degrees, four actual corners, error={max_error:.6f} m")
        verifier = CornerMotionVerifier.create(initial_positions_w)
        markers = create_corner_markers()
        marker_indices = torch.arange(4, dtype=torch.long, device=env.unwrapped.device)
        hold_action = make_hold_action(env)

        print(f"[corner_tracker] node count: {initial_nodes_w.shape[0]}")
        for label, index in zip(CORNER_LABELS, corner_indices.detach().cpu().tolist(), strict=True):
            print(f"[corner_tracker] {label}: vertex index {index}")
        print(f"[corner_tracker] step 0 | {format_positions(initial_positions_w - env_origin)}")

        csv_file = None
        csv_writer = None
        try:
            if args_cli.csv:
                csv_path = Path(args_cli.csv).expanduser().resolve()
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_file = csv_path.open("w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(("step", "label", "vertex_index", "x", "y", "z", "displacement_m"))

            with torch.inference_mode():
                for step in range(1, args_cli.num_steps + 1):
                    env.step(hold_action)
                    positions_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
                    displacement = verifier.update(positions_w)
                    markers.visualize(translations=positions_w, marker_indices=marker_indices)

                    if csv_writer is not None:
                        local_positions = (positions_w - env_origin).detach().cpu().tolist()
                        displacement_values = displacement.detach().cpu().tolist()
                        for label, index, xyz, distance in zip(
                            CORNER_LABELS,
                            corner_indices.detach().cpu().tolist(),
                            local_positions,
                            displacement_values,
                            strict=True,
                        ):
                            csv_writer.writerow((step, label, index, *xyz, distance))

                    if step % args_cli.print_interval == 0 or step == args_cli.num_steps:
                        print(
                            f"[corner_tracker] step {step} | "
                            f"{format_positions(positions_w - env_origin)} | "
                            f"displacement={displacement.detach().cpu().tolist()}"
                        )
        finally:
            if csv_file is not None:
                csv_file.close()
            env.close()

        moved = verifier.moved_mask(args_cli.movement_threshold).detach().cpu().tolist()
        max_displacements = verifier.max_displacements.detach().cpu().tolist()
        for label, did_move, distance in zip(CORNER_LABELS, moved, max_displacements, strict=True):
            status = "PASS" if did_move else "CHECK"
            print(f"[corner_tracker] {status} {label}: max displacement={distance:.6f} m")
        if all(moved):
            print("[corner_tracker] PASS: all four fixed vertex indices moved with the cloth.")
        else:
            print(
                "[corner_tracker] CHECK: one or more corners stayed below the movement threshold. "
                "Move or pull the cloth and rerun, or lower --movement_threshold only if justified."
            )


if __name__ == "__main__":
    main()

