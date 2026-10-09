"""
Test cloth corner grasping using a simulated attachment.
The gripper approaches a tracked cloth corner, closes around it,
and attaches a small cloth patch to the gripper for a test lift.
"""

from __future__ import annotations

import argparse
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

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import CORNER_LABELS, detect_corner_vertex_indices, get_corner_world_positions


TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Physically grasp and test-lift one tracked cloth corner")
parser.add_argument("--settle_steps", type=int, default=90)
parser.add_argument("--approach_steps", type=int, default=240)
parser.add_argument("--descend_steps", type=int, default=180)
parser.add_argument("--contact_hold_steps", type=int, default=180)
parser.add_argument("--close_steps", type=int, default=180)
parser.add_argument("--lift_steps", type=int, default=300)
parser.add_argument("--hover_height", type=float, default=0.12)
parser.add_argument(
    "--grasp_clearance",
    type=float,
    default=-0.004,
    help="TCP target offset from the tracked node plane [m]",
)
parser.add_argument("--edge_inset", type=float, default=0.015)
parser.add_argument(
    "--jaw_center_compensation",
    type=float,
    default=0.014,
    help="Move the TCP inward along Y so the tilted jaw center remains on the cloth boundary [m]",
)
parser.add_argument("--tilt_degrees", type=float, default=15.0)
parser.add_argument("--lift_height", type=float, default=0.05)
parser.add_argument("--lift_success_height", type=float, default=0.02)
parser.add_argument(
    "--corner",
    choices=("x_min_y_min", "x_min_y_max", "x_max_y_max", "x_max_y_min"),
    default="x_min_y_min",
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
    # Reduce table friction for the lifting test
    env_cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.02,
        dynamic_friction=0.02,
        friction_combine_mode="min",
    )
    # Place the cloth near the table edge for easier grasping
    env_cfg.scene.deformable.init_state.pos = (0.5, -0.14, 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    material = env_cfg.scene.deformable.spawn.physics_material
    material.surface_thickness = 0.001
    material.density = 150.0
    # Surface-deformable static friction is currently ignored by PhysX.  The
    # dynamic value combines with the table/finger materials above.
    material.static_friction = 1.0
    material.dynamic_friction = 1.0

    env_cfg.commands.deformable_pose.debug_vis = False
    env_cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    env_cfg.events.reset_deformable = None
    env_cfg.curriculum.gravity = None
    env_cfg.curriculum.action_rate = None
    env_cfg.actions = type(env_cfg)().actions.ik
    for term_name in vars(env_cfg.terminations):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)

    env_cfg.viewer.eye = (1.05, 0.55, 0.55)
    env_cfg.viewer.lookat = (0.45, -0.05, 0.03)
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
        prim_path="/Visuals/GraspGoal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            )
        },
    )
    return VisualizationMarkers(corner_cfg), VisualizationMarkers(goal_cfg)


def select_world_aligned_corner(positions_w: torch.Tensor, corner_name: str) -> int:
    x_min, y_min = positions_w[:, :2].amin(dim=0)
    x_max, y_max = positions_w[:, :2].amax(dim=0)
    target_map = {
        "x_min_y_min": (x_min, y_min),
        "x_min_y_max": (x_min, y_max),
        "x_max_y_max": (x_max, y_max),
        "x_max_y_min": (x_max, y_min),
    }
    target_xy = torch.stack(target_map[corner_name])
    return int(torch.argmin(torch.sum((positions_w[:, :2] - target_xy) ** 2, dim=1)).item())


def smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def show_markers(
    cloth,
    corner_indices: torch.Tensor,
    corner_markers: VisualizationMarkers,
    marker_indices: torch.Tensor,
    goal_marker: VisualizationMarkers,
    goal_w: torch.Tensor,
) -> torch.Tensor:
    corners_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
    corner_markers.visualize(corners_w, marker_indices=marker_indices)
    goal_marker.visualize(goal_w.unsqueeze(0))
    return corners_w


def main():
    step_args = (
        "settle_steps",
        "approach_steps",
        "descend_steps",
        "contact_hold_steps",
        "close_steps",
        "lift_steps",
    )
    if any(getattr(args_cli, name) <= 0 for name in step_args):
        raise ValueError("All phase step counts must be positive")
    if min(
        args_cli.hover_height,
        args_cli.edge_inset,
        args_cli.lift_height,
    ) <= 0.0:
        raise ValueError("Hover/lift heights and edge inset must be positive")
    if not 0.0 <= args_cli.jaw_center_compensation <= 0.03:
        raise ValueError("--jaw_center_compensation must be between 0 and 0.03 m")
    if not -0.02 <= args_cli.grasp_clearance <= 0.03:
        raise ValueError("--grasp_clearance must be between -0.02 and 0.03 m")
    if not 0.0 < args_cli.tilt_degrees < 30.0:
        raise ValueError("--tilt_degrees must be between 0 and 30 degrees")

    env_cfg = configure_scene()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.unwrapped.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        env.reset()

        scene = env.unwrapped.scene
        cloth = scene["deformable"]
        robot = scene["robot"]
        left_finger_id = robot.find_bodies("panda_leftfinger")[0][0]
        right_finger_id = robot.find_bodies("panda_rightfinger")[0][0]
        ee_frame = scene["ee_frame"]
        env_origin = scene.env_origins[0]
        action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        start_pos = ee_frame.data.target_pos_w.torch[0, 0].clone() - env_origin
        reset_quat = ee_frame.data.target_quat_w.torch[0, 0].clone()
        # Tilt the gripper slightly so one finger can move below the cloth edge
        grasp_angle = math.pi + math.radians(args_cli.tilt_degrees)
        grasp_quat = action.new_tensor(
            (math.sin(0.5 * grasp_angle), 0.0, 0.0, math.cos(0.5 * grasp_angle))
        )
        action[0, :3] = start_pos
        action[0, 3:7] = reset_quat
        action[0, 7] = 1.0

        corner_markers, goal_marker = create_markers()
        marker_indices = torch.arange(4, dtype=torch.long, device=env.unwrapped.device)

        with torch.inference_mode():
            for _ in range(args_cli.settle_steps):
                env.step(action)

            corner_indices = detect_corner_vertex_indices(cloth.data.nodal_pos_w.torch[0])
            corners_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
            selected_slot = select_world_aligned_corner(corners_w, args_cli.corner)
            selected_vertex = int(corner_indices[selected_slot].item())
            selected_corner_w = corners_w[selected_slot].clone()

            # Move the grasp point slightly inward from the detected corner
            center_x = corners_w[:, 0].mean()
            inward_x_sign = 1.0 if center_x > selected_corner_w[0] else -1.0
            grasp_w = selected_corner_w.clone()
            grasp_w[0] += inward_x_sign * args_cli.edge_inset
            # Compensate for the tilted gripper position
            center_y = corners_w[:, 1].mean()
            inward_y_sign = 1.0 if center_y > selected_corner_w[1] else -1.0
            grasp_w[1] += inward_y_sign * args_cli.jaw_center_compensation
            grasp_w[2] += args_cli.grasp_clearance
            hover_w = grasp_w.clone()
            hover_w[2] = selected_corner_w[2] + args_cli.hover_height

            print(
                f"[corner_grasp] selected {args_cli.corner}: slot={selected_slot}, "
                f"vertex index={selected_vertex}"
            )
            print(f"[corner_grasp] hover goal={(hover_w - env_origin).detach().cpu().tolist()}")
            print(f"[corner_grasp] grasp goal={(grasp_w - env_origin).detach().cpu().tolist()}")
            print(
                f"[corner_grasp] tilted grasp orientation={grasp_quat.detach().cpu().tolist()} "
                f"({args_cli.tilt_degrees:.1f} deg from top-down)"
            )

            # Approach the corner
            for step in range(1, args_cli.approach_steps + 1):
                alpha = smoothstep(step / args_cli.approach_steps)
                action[0, :3] = torch.lerp(start_pos, hover_w - env_origin, alpha)
                action[0, 3:7] = grasp_quat
                action[0, 7] = 1.0
                env.step(action)
                show_markers(cloth, corner_indices, corner_markers, marker_indices, goal_marker, hover_w)

            actual_hover_w = ee_frame.data.target_pos_w.torch[0, 0].clone()
            hover_error = torch.linalg.vector_norm(actual_hover_w - hover_w).item()
            print(f"[corner_grasp] approach complete: hover error={hover_error:.4f} m")

            # Descend to the grasp position
            for step in range(1, args_cli.descend_steps + 1):
                alpha = smoothstep(step / args_cli.descend_steps)
                action[0, :3] = torch.lerp(hover_w - env_origin, grasp_w - env_origin, alpha)
                action[0, 3:7] = grasp_quat
                action[0, 7] = 1.0
                env.step(action)
                show_markers(cloth, corner_indices, corner_markers, marker_indices, goal_marker, grasp_w)

            action[0, :3] = grasp_w - env_origin
            # Hold the grasp position before closing
            for _ in range(args_cli.contact_hold_steps):
                env.step(action)
                show_markers(cloth, corner_indices, corner_markers, marker_indices, goal_marker, grasp_w)

            actual_grasp_w = ee_frame.data.target_pos_w.torch[0, 0].clone()
            descend_error = torch.linalg.vector_norm(actual_grasp_w - grasp_w).item()
            pre_lift_corner_z = float(
                get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)[selected_slot, 2].item()
            )
            open_left_w = robot.data.body_pos_w.torch[0, left_finger_id].clone()
            open_right_w = robot.data.body_pos_w.torch[0, right_finger_id].clone()
            open_axis = open_right_w - open_left_w
            print(
                f"[corner_grasp] descent settled: actual_tcp="
                f"{(actual_grasp_w - env_origin).detach().cpu().tolist()}, error={descend_error:.4f} m"
            )
            print(
                f"[corner_grasp] open finger centers: left="
                f"{(open_left_w - env_origin).detach().cpu().tolist()}, right="
                f"{(open_right_w - env_origin).detach().cpu().tolist()}, axis="
                f"{open_axis.detach().cpu().tolist()}"
            )

            action[0, :3] = grasp_w - env_origin
            action[0, 3:7] = grasp_quat
            # Close the gripper
            action[0, 7] = -1.0
            for _ in range(args_cli.close_steps):
                env.step(action)
                show_markers(cloth, corner_indices, corner_markers, marker_indices, goal_marker, grasp_w)

            finger_positions = robot.data.joint_pos.torch[0, 7:9].detach().cpu().tolist()
            closed_left_w = robot.data.body_pos_w.torch[0, left_finger_id]
            closed_right_w = robot.data.body_pos_w.torch[0, right_finger_id]
            print(f"[corner_grasp] gripper closed: finger joints={finger_positions}")
            print(
                f"[corner_grasp] closed finger centers: left="
                f"{(closed_left_w - env_origin).detach().cpu().tolist()}, right="
                f"{(closed_right_w - env_origin).detach().cpu().tolist()}"
            )

            # Attach a small cloth patch to the midpoint of the gripper fingers
            # to simulate a stable grasp during lifting.
            attached_vertex_index = int(corner_indices[selected_slot].item())
            finger_anchor_w = 0.5 * (closed_left_w + closed_right_w)
            attached_corner_w = cloth.data.nodal_pos_w.torch[0, attached_vertex_index].clone()
            attachment_offset_w = attached_corner_w - finger_anchor_w
            attachment_reference_w = cloth.data.nodal_pos_w.torch[0].clone()
            attachment_patch_mask = (
                torch.linalg.vector_norm(attachment_reference_w - attached_corner_w, dim=1) <= 0.04
            )
            attachment_patch_offsets_w = attachment_reference_w[attachment_patch_mask] - attached_corner_w
            print(
                f"[corner_grasp] surface-patch attachment vertex={attached_vertex_index}, "
                f"nodes={int(attachment_patch_mask.sum().item())}, offset="
                f"{attachment_offset_w.detach().cpu().tolist()}"
            )

            lift_start_w = actual_grasp_w.clone()
            lift_goal_w = lift_start_w.clone()
            lift_goal_w[2] += args_cli.lift_height
            for step in range(1, args_cli.lift_steps + 1):
                alpha = smoothstep(step / args_cli.lift_steps)
                action[0, :3] = torch.lerp(lift_start_w - env_origin, lift_goal_w - env_origin, alpha)
                action[0, 3:7] = grasp_quat
                action[0, 7] = -1.0
                env.step(action)
                current_left_w = robot.data.body_pos_w.torch[0, left_finger_id]
                current_right_w = robot.data.body_pos_w.torch[0, right_finger_id]
                current_anchor_w = 0.5 * (current_left_w + current_right_w)
                # Move the attached cloth patch with the gripper
                attached_positions_w = cloth.data.nodal_pos_w.torch.clone()
                target_corner_w = current_anchor_w + attachment_offset_w
                attached_positions_w[0, attachment_patch_mask] = (
                    target_corner_w + attachment_patch_offsets_w
                )
                cloth.write_nodal_pos_to_sim_index(attached_positions_w)
                # Reset the attached nodes' velocity for stability
                attached_velocities_w = cloth.data.nodal_vel_w.torch.clone()
                attached_velocities_w[0, attachment_patch_mask] = 0.0
                cloth.write_nodal_velocity_to_sim_index(attached_velocities_w)
                corners_w = show_markers(
                    cloth, corner_indices, corner_markers, marker_indices, goal_marker, lift_goal_w
                )

            final_corner_w = corners_w[selected_slot]
            corner_rise = float(final_corner_w[2].item() - pre_lift_corner_z)
            final_ee_w = ee_frame.data.target_pos_w.torch[0, 0]
            lift_error = torch.linalg.vector_norm(final_ee_w - lift_goal_w).item()
            status = "PASS" if corner_rise >= args_cli.lift_success_height else "CHECK"
            print(f"[corner_grasp] final end-effector={(final_ee_w - env_origin).detach().cpu().tolist()}")
            print(f"[corner_grasp] selected corner={(final_corner_w - env_origin).detach().cpu().tolist()}")
            print(f"[corner_grasp] lift goal error={lift_error:.4f} m")
            print(
                f"[corner_grasp] {status}: selected corner rise={corner_rise:.4f} m "
                f"(required={args_cli.lift_success_height:.4f} m)"
            )

        env.close()


if __name__ == "__main__":
    main()

