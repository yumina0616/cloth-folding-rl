"""
Move the Franka gripper above a tracked cloth corner.
The target corner is selected from the four tracked cloth corners.
The end-effector approaches and holds a position above the selected corner.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401 -- registers tasks

from cloth_corner_tracker import CORNER_LABELS, detect_corner_vertex_indices, get_corner_world_positions


TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Approach a tracked cloth corner with the Franka gripper")
parser.add_argument("--settle_steps", type=int, default=90)
parser.add_argument("--approach_steps", type=int, default=240)
parser.add_argument("--hold_steps", type=int, default=120)
parser.add_argument("--hover_height", type=float, default=0.12, help="Height above the live corner [m]")
parser.add_argument("--position_tolerance", type=float, default=0.03, help="Final end-effector error threshold [m]")
parser.add_argument(
    "--corner",
    choices=("x_min_y_min", "x_min_y_max", "x_max_y_max", "x_max_y_min"),
    default="x_min_y_min",
    help="Corner selected in the environment's world-aligned XY plane",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def configure_scene():
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0
    env_cfg.scene.support_neg_y = None
    env_cfg.scene.support_pos_y = None
    env_cfg.scene.table.spawn.size = (1.3, 0.5, 1.05)
    env_cfg.scene.table.spawn.visible = True

    # Place the cloth at the center of the table
    env_cfg.scene.deformable.init_state.pos = (0.5, 0.0, 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
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

    env_cfg.viewer.eye = (1.1, 0.65, 0.65)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.05)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)
    return env_cfg


def create_markers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    colors = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.4, 1.0), (1.0, 0.8, 0.1))
    corner_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ClothCorners",
        markers={
            label: sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for label, color in zip(CORNER_LABELS, colors, strict=True)
        },
    )
    goal_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CornerApproachGoal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            )
        },
    )
    return VisualizationMarkers(corner_cfg), VisualizationMarkers(goal_cfg)


def select_world_aligned_corner(positions_w: torch.Tensor, corner_name: str) -> int:
    """Select the tracked point nearest to a requested XY bounding-box corner."""
    x_min, y_min = positions_w[:, :2].amin(dim=0)
    x_max, y_max = positions_w[:, :2].amax(dim=0)
    targets = {
        "x_min_y_min": (x_min, y_min),
        "x_min_y_max": (x_min, y_max),
        "x_max_y_max": (x_max, y_max),
        "x_max_y_min": (x_max, y_min),
    }
    target_xy = torch.stack(targets[corner_name])
    return int(torch.argmin(torch.sum((positions_w[:, :2] - target_xy) ** 2, dim=1)).item())


def make_hold_action(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ee_frame = env.unwrapped.scene["ee_frame"]
    env_origin = env.unwrapped.scene.env_origins[0]
    start_position = ee_frame.data.target_pos_w.torch[0, 0].clone() - env_origin
    start_orientation = ee_frame.data.target_quat_w.torch[0, 0].clone()
    action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    action[0, :3] = start_position
    action[0, 3:7] = start_orientation
    action[0, 7] = 1.0
    return action, start_position, start_orientation


def smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def main():
    for name in ("settle_steps", "approach_steps", "hold_steps"):
        if getattr(args_cli, name) < 0:
            raise ValueError(f"--{name} cannot be negative")
    if args_cli.approach_steps == 0:
        raise ValueError("--approach_steps must be positive")
    if args_cli.hover_height <= 0.0:
        raise ValueError("--hover_height must be positive")

    env_cfg = configure_scene()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.unwrapped.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        env.reset()

        cloth = env.unwrapped.scene["deformable"]
        ee_frame = env.unwrapped.scene["ee_frame"]
        env_origin = env.unwrapped.scene.env_origins[0]
        action, start_position, start_orientation = make_hold_action(env)
        corner_markers, goal_marker = create_markers()
        marker_indices = torch.arange(4, dtype=torch.long, device=env.unwrapped.device)

        with torch.inference_mode():
            for _ in range(args_cli.settle_steps):
                env.step(action)

            # Detect the four cloth corners after settling
            nodes_w = cloth.data.nodal_pos_w.torch[0]
            corner_indices = detect_corner_vertex_indices(nodes_w)
            corner_positions_w = get_corner_world_positions(nodes_w, corner_indices)
            # Select the target corner
            selected_slot = select_world_aligned_corner(corner_positions_w, args_cli.corner)
            selected_vertex = int(corner_indices[selected_slot].item())
            # Use a top-down orientation for the approach
            approach_orientation = action.new_tensor((1.0, 0.0, 0.0, 0.0))

            print(
                f"[corner_approach] selected {args_cli.corner}: "
                f"tracked slot={selected_slot}, vertex index={selected_vertex}"
            )
            print(
                f"[corner_approach] hover height={args_cli.hover_height:.3f} m, "
                f"approach steps={args_cli.approach_steps}"
            )

            # Move smoothly toward the live-tracked corner
            for step in range(1, args_cli.approach_steps + 1):
                corner_positions_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
                live_goal_w = corner_positions_w[selected_slot].clone()
                live_goal_w[2] += args_cli.hover_height
                live_goal_local = live_goal_w - env_origin
                alpha = smoothstep(step / args_cli.approach_steps)
                action[0, :3] = torch.lerp(start_position, live_goal_local, alpha)
                action[0, 3:7] = approach_orientation
                action[0, 7] = 1.0
                env.step(action)
                corner_markers.visualize(corner_positions_w, marker_indices=marker_indices)
                goal_marker.visualize(live_goal_w.unsqueeze(0))

                if step in (1, args_cli.approach_steps // 2, args_cli.approach_steps):
                    actual_ee_w = ee_frame.data.target_pos_w.torch[0, 0]
                    error = torch.linalg.vector_norm(actual_ee_w - live_goal_w).item()
                    print(f"[corner_approach] approach step {step}: ee-to-goal error={error:.4f} m")

            for _ in range(args_cli.hold_steps):
                corner_positions_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
                live_goal_w = corner_positions_w[selected_slot].clone()
                live_goal_w[2] += args_cli.hover_height
                action[0, :3] = live_goal_w - env_origin
                env.step(action)
                corner_markers.visualize(corner_positions_w, marker_indices=marker_indices)
                goal_marker.visualize(live_goal_w.unsqueeze(0))

            actual_ee_w = ee_frame.data.target_pos_w.torch[0, 0]
            final_error = torch.linalg.vector_norm(actual_ee_w - live_goal_w).item()
            final_ee = (actual_ee_w - env_origin).detach().cpu().tolist()
            final_goal = (live_goal_w - env_origin).detach().cpu().tolist()
            status = "PASS" if final_error <= args_cli.position_tolerance else "CHECK"
            print(f"[corner_approach] final end-effector={final_ee}")
            print(f"[corner_approach] final hover goal={final_goal}")
            print(
                f"[corner_approach] {status}: final error={final_error:.4f} m "
                f"(tolerance={args_cli.position_tolerance:.4f} m)"
            )

        env.close()


if __name__ == "__main__":
    main()

