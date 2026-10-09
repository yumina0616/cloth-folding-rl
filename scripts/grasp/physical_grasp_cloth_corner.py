"""
Test physical grasping of a cloth corner.
The gripper approaches a tracked cloth corner, closes around the cloth,
and attempts to lift it using physical contact without node attachment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions


TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Physical-only cloth corner grasp")
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--push_distance", type=float, default=0.100)
parser.add_argument("--lift_height", type=float, default=0.060)
parser.add_argument("--lift_success_height", type=float, default=0.020)
parser.add_argument("--hold_steps", type=int, default=180)
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

    # Table top is z=0 and its near edge is y=-0.25 m.
    env_cfg.scene.table.spawn.size = (1.3, 0.5, 1.05)
    env_cfg.scene.table.spawn.visible = True
    env_cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.04,
        dynamic_friction=0.03,
        restitution=0.0,
        friction_combine_mode="min",
    )

    # Place the cloth near the table edge for easier grasping
    env_cfg.scene.deformable.init_state.pos = (0.5, -0.14, 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cloth_material = env_cfg.scene.deformable.spawn.physics_material
    cloth_material.surface_thickness = 0.006
    cloth_material.density = 150.0
    cloth_material.static_friction = 2.0
    cloth_material.dynamic_friction = 1.5
    cloth_material.elasticity_damping = 0.25
    cloth_material.bend_damping = 0.25

    # Increase gripper friction for physical cloth grasping
    env_cfg.scene.robot.spawn.make_uninstanceable = True
    env_cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=1.5,
        restitution=0.0,
        friction_combine_mode="max",
    )
    # Use a more compliant gripper actuator for cloth contact
    env_cfg.scene.robot.actuators["panda_hand"].joint_effort_limit = 70.0
    env_cfg.scene.robot.actuators["panda_hand"].stiffness = 500.0
    env_cfg.scene.robot.actuators["panda_hand"].damping = 100.0

    env_cfg.commands.deformable_pose.debug_vis = False
    env_cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    env_cfg.events.reset_deformable = None
    env_cfg.curriculum.gravity = None
    env_cfg.curriculum.action_rate = None
    env_cfg.actions = type(env_cfg)().actions.ik
    for term_name in vars(env_cfg.terminations):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)

    env_cfg.viewer.eye = (1.05, 0.55, 0.48)
    env_cfg.viewer.lookat = (0.48, -0.16, 0.03)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)
    return env_cfg


def smoothstep(value: float) -> float:
    """Smooth interpolation for robot motion."""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def nearest_corner_slot(corners_w: torch.Tensor) -> int:
    """Return the corner with minimum x and y coordinates"""
    target_xy = torch.stack((corners_w[:, 0].min(), corners_w[:, 1].min()))
    return int(torch.argmin(torch.sum((corners_w[:, :2] - target_xy) ** 2, dim=1)).item())


def main():
    if args_cli.push_distance <= 0.01:
        raise ValueError("--push_distance must exceed the initial 10 mm table margin")
    if args_cli.lift_height < args_cli.lift_success_height:
        raise ValueError("--lift_height must be at least --lift_success_height")

    env_cfg = configure_scene()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.unwrapped.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        env.reset()

        scene = env.unwrapped.scene
        cloth = scene["deformable"]
        robot = scene["robot"]
        ee_frame = scene["ee_frame"]
        env_origin = scene.env_origins[0]
        left_finger_id = robot.find_bodies("panda_leftfinger")[0][0]
        right_finger_id = robot.find_bodies("panda_rightfinger")[0][0]
        finger_joint_ids = robot.find_joints("panda_finger_joint.*")[0]

        action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        command_pos = ee_frame.data.target_pos_w.torch[0, 0].clone() - env_origin
        command_quat = ee_frame.data.target_quat_w.torch[0, 0].clone()
        action[0, :3] = command_pos
        action[0, 3:7] = command_quat
        action[0, 7] = 1.0

        top_down_quat = action.new_tensor((1.0, 0.0, 0.0, 0.0))
        # Tool axis points toward +Y; the jaws close mainly along world Z.
        side_quat = action.new_tensor((-0.70710678, 0.0, 0.0, 0.70710678))

        def hold(steps: int, gripper: float) -> None:
            action[0, :3] = command_pos
            action[0, 3:7] = command_quat
            action[0, 7] = gripper
            for _ in range(steps):
                env.step(action)

        def move_to(goal_w: torch.Tensor, quat: torch.Tensor, steps: int, gripper: float) -> None:
            nonlocal command_pos, command_quat
            start_pos = command_pos.clone()
            goal_pos = goal_w - env_origin
            # Smoothly interpolate the end-effector orientation
            start_quat = command_quat.clone()
            for step in range(1, steps + 1):
                alpha = smoothstep(step / steps)
                action[0, :3] = torch.lerp(start_pos, goal_pos, alpha)
                blended_quat = torch.lerp(start_quat, quat, alpha)
                action[0, 3:7] = blended_quat / torch.linalg.vector_norm(blended_quat)
                action[0, 7] = gripper
                env.step(action)
            command_pos = goal_pos.clone()
            command_quat = quat.clone()

        with torch.inference_mode():
            # Let the cloth settle before detecting its corners
            hold(args_cli.settle_steps, 1.0)
            # Detect and select the target cloth corner
            corner_indices = detect_corner_vertex_indices(cloth.data.nodal_pos_w.torch[0])
            corners_w = get_corner_world_positions(cloth.data.nodal_pos_w.torch[0], corner_indices)
            selected_slot = nearest_corner_slot(corners_w)
            selected_vertex = int(corner_indices[selected_slot].item())
            initial_corner_w = corners_w[selected_slot].clone()
            table_edge_y = -0.25 + float(env_origin[1].item())
            initial_on_table = bool(initial_corner_w[1].item() >= table_edge_y)
            print(
                f"[physical_grasp] selected slot={selected_slot}, vertex={selected_vertex}, "
                f"initial={(initial_corner_w - env_origin).cpu().tolist()}, all-corners-on-table={initial_on_table}"
            )

            # Phase 1: push the selected corner over the table edge
            center_x = corners_w[:, 0].mean()
            inward_x = 1.0 if center_x > initial_corner_w[0] else -1.0
            press_w = initial_corner_w.clone()
            press_w[0] += inward_x * 0.015
            press_w[1] += 0.015
            press_w[2] += 0.055
            move_to(press_w, top_down_quat, 220, 1.0)
            hold(90, -1.0)
            press_w[2] = initial_corner_w[2] + 0.001
            move_to(press_w, top_down_quat, 160, -1.0)
            hold(60, -1.0)
            pushed_w = press_w.clone()
            pushed_w[1] -= args_cli.push_distance
            move_to(pushed_w, top_down_quat, 300, -1.0)
            hold(60, -1.0)
            pushed_w[2] += 0.065
            move_to(pushed_w, top_down_quat, 160, 1.0)

            overhang_corner_w = cloth.data.nodal_pos_w.torch[0, selected_vertex].clone()
            overhang = table_edge_y - float(overhang_corner_w[1].item())
            print(
                f"[physical_grasp] after edge-slide corner="
                f"{(overhang_corner_w - env_origin).cpu().tolist()}, overhang={overhang:.4f} m"
            )

            # Phase 2: approach and pinch the overhanging cloth
            outside_w = overhang_corner_w.clone()
            outside_w[1] -= 0.060
            outside_w[2] += 0.055
            move_to(outside_w, side_quat, 220, 1.0)
            outside_w[2] = max(float(overhang_corner_w[2].item()), 0.004 + float(env_origin[2].item()))
            move_to(outside_w, side_quat, 160, 1.0)
            pinch_w = outside_w.clone()
            pinch_w[1] = overhang_corner_w[1] - 0.004
            move_to(pinch_w, side_quat, 220, 1.0)
            hold(220, -1.0)

            closed_corner_w = cloth.data.nodal_pos_w.torch[0, selected_vertex].clone()
            left_w = robot.data.body_pos_w.torch[0, left_finger_id].clone()
            right_w = robot.data.body_pos_w.torch[0, right_finger_id].clone()
            closed_anchor_w = 0.5 * (left_w + right_w)
            closed_distance = torch.linalg.vector_norm(closed_corner_w - closed_anchor_w).item()
            finger_positions = robot.data.joint_pos.torch[0, finger_joint_ids].cpu().tolist()
            print(
                f"[physical_grasp] pinch finger-joints={finger_positions}, "
                f"corner-to-jaw-center={closed_distance:.4f} m"
            )

            # Retract from the table edge before lifting
            retract_w = pinch_w.clone()
            retract_w[1] -= 0.020
            move_to(retract_w, side_quat, 160, -1.0)
            lift_w = retract_w.clone()
            lift_w[2] += args_cli.lift_height
            move_to(lift_w, side_quat, 360, -1.0)

            minimum_rise = float("inf")
            maximum_follow_distance = 0.0
            for _ in range(args_cli.hold_steps):
                hold(1, -1.0)
                corner_w = cloth.data.nodal_pos_w.torch[0, selected_vertex]
                left_w = robot.data.body_pos_w.torch[0, left_finger_id]
                right_w = robot.data.body_pos_w.torch[0, right_finger_id]
                anchor_w = 0.5 * (left_w + right_w)
                minimum_rise = min(minimum_rise, float(corner_w[2].item() - initial_corner_w[2].item()))
                maximum_follow_distance = max(
                    maximum_follow_distance,
                    torch.linalg.vector_norm(corner_w - anchor_w).item(),
                )

            final_corner_w = cloth.data.nodal_pos_w.torch[0, selected_vertex].clone()
            final_rise = float(final_corner_w[2].item() - initial_corner_w[2].item())
            # Evaluate physical grasp success without virtual attachment
            passed = (
                initial_on_table
                and overhang >= 0.010
                and minimum_rise >= args_cli.lift_success_height
                and maximum_follow_distance <= 0.050
            )
            status = "PASS" if passed else "CHECK"
            print(f"[physical_grasp] final corner={(final_corner_w - env_origin).cpu().tolist()}")
            print(
                f"[physical_grasp] {status}: final-rise={final_rise:.4f} m, "
                f"hold-min-rise={minimum_rise:.4f} m, max-follow-distance={maximum_follow_distance:.4f} m"
            )

        env.close()


if __name__ == "__main__":
    main()
