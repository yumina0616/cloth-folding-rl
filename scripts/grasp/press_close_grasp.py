"""
Physical cloth grasp by pressing and closing (experiment 006).
The fingertips press the cloth near a corner and close,
so that friction gathers the cloth into a ridge between the fingers.
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

import isaaclab_tasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions

TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Press-and-close physical cloth grasp")
parser.add_argument("--press_depth", type=float, default=0.0, help="Fingertip target below the cloth height [m] (negative = above the cloth)")
parser.add_argument("--inset", type=float, default=0.02, help="Grasp point distance from the corner [m]")
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=220)
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

    # Same physics as physical_grasp_cloth_corner.py (baseline)
    env_cfg.scene.table.spawn.size = (1.3, 0.5, 1.05)
    env_cfg.scene.table.spawn.visible = True
    env_cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.04,
        dynamic_friction=0.03,
        restitution=0.0,
        friction_combine_mode="min",
    )

    # [006] Place the cloth at the table center (no table edge needed)
    env_cfg.scene.deformable.init_state.pos = (0.5, 0.0, 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cloth_material = env_cfg.scene.deformable.spawn.physics_material
    cloth_material.surface_thickness = 0.006
    cloth_material.density = 150.0
    cloth_material.static_friction = 2.0
    cloth_material.elasticity_damping = 0.25
    cloth_material.dynamic_friction = 1.5
    cloth_material.bend_damping = 0.25

    env_cfg.scene.robot.spawn.make_uninstanceable = True
    env_cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=1.5,
        restitution=0.0,
        friction_combine_mode="max",
    )
    env_cfg.scene.robot.actuators["panda_hand"].joint_effort_limit = 70.0
    env_cfg.scene.robot.actuators["panda_hand"].stiffness =500.0
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

    env_cfg.viewer.eye = (1.05, 0.45, 0.40)
    env_cfg.viewer.lookat =  (0.45, -0.05, 0.02)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)

    return env_cfg

def smoothstep(value: float) -> float:
    """Smooth interpolation for robot motion."""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def nearest_corner_slot(corners_w: torch.Tensor) -> int:
    """Return the corner with minimum x and y coordinates."""
    target_xy = torch.stack((corners_w[:, 0].min(), corners_w[:, 1].min()))
    return int(torch.argmin(torch.sum((corners_w[:, :2] - target_xy) ** 2, dim=1)).item())


def main():
    env_cfg = configure_scene()
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK, cfg=env_cfg)
        env.unwrapped.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        env.reset()

        scene = env.unwrapped.scene
        cloth = scene["deformable"]
        ee_frame = scene["ee_frame"]
        robot = scene["robot"]
        left_finger_id = robot.find_bodies("panda_leftfinger")[0][0]
        right_finger_id = robot.find_bodies("panda_rightfinger")[0][0]
        finger_joint_ids = robot.find_joints("panda_finger_joint.*")[0]
        env_origin = scene.env_origins[0]

        action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        command_pos = ee_frame.data.target_pos_w.torch[0, 0].clone() - env_origin
        command_quat = ee_frame.data.target_quat_w.torch[0, 0].clone()
        top_down_quat = action.new_tensor((1.0, 0.0, 0.0, 0.0))

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
            hold(args_cli.settle_steps, 1.0)
            nodes_w = cloth.data.nodal_pos_w.torch[0]
            corner_indices = detect_corner_vertex_indices(nodes_w)
            corners_w = get_corner_world_positions(nodes_w, corner_indices)
            slot = nearest_corner_slot(corners_w)
            corner_w = corners_w[slot].clone()

            center_xy = corners_w[:, :2].mean(dim=0)
            inward = center_xy - corner_w[:2]
            inward = inward / torch.linalg.vector_norm(inward)
            grasp_xy = corner_w[:2] + args_cli.inset * inward
            grasp_node = int(torch.argmin(torch.linalg.vector_norm(nodes_w[:, :2] - grasp_xy, dim=1)).item())
            
            cloth_z = float(nodes_w[grasp_node, 2].item())


            above_w = torch.cat((grasp_xy, grasp_xy.new_tensor([cloth_z + 0.055])))
            move_to(above_w, top_down_quat, 220, 1.0)

            
            press_w = above_w.clone()
            press_w[2] = cloth_z - args_cli.press_depth
            move_to(press_w, top_down_quat, 160, 1.0)
            hold(60, 1.0)

            # State right before closing
            ee_z = float(ee_frame.data.target_pos_w.torch[0, 0, 2].item())
            node_z = float(cloth.data.nodal_pos_w.torch[0, grasp_node, 2].item())
            press_gap = ee_z - node_z
            left_w = robot.data.body_pos_w.torch[0, left_finger_id]
            right_w = robot.data.body_pos_w.torch[0, right_finger_id]
            jaw_axis = (right_w - left_w)[:2]
            jaw_axis = jaw_axis / torch.linalg.vector_norm(jaw_axis)
            print(f"[press_close] jaw axis (xy)={jaw_axis.cpu().tolist()}, diagonal (xy)={inward.cpu().tolist()}")

            # Close while holding the height and track the ridge between the fingers
            max_ridge = float("-inf")
            for _ in range(args_cli.close_steps):
                hold(1, -1.0)
                nodes_now = cloth.data.nodal_pos_w.torch[0]
                near = torch.linalg.vector_norm(nodes_now[:, :2] - grasp_xy, dim=1) <= 0.03
                ridge = float(nodes_now[near, 2].max().item()) - cloth_z
                max_ridge = max(max_ridge, ridge)
            finger_gap = float(robot.data.joint_pos.torch[0, finger_joint_ids].sum().item())

            # Lift and measure how high the grasped point stays
            lift_w = press_w.clone()
            lift_w[2] += args_cli.lift_height
            move_to(lift_w, top_down_quat, 360, -1.0)
            min_rise = float("inf")
            for _ in range(args_cli.hold_steps):
                hold(1, -1.0)
                rise = float(cloth.data.nodal_pos_w.torch[0, grasp_node, 2].item()) - cloth_z
                min_rise = min(min_rise, rise)
            corner_index = int(corner_indices[slot].item())
            corner_rise = float(cloth.data.nodal_pos_w.torch[0, corner_index, 2].item() - corner_w[2].item())

            passed = min_rise >= args_cli.lift_success_height
            print(
                f"[press_close] SUMMARY | depth={args_cli.press_depth * 1000:.1f} mm"
                f" | press_gap={press_gap * 1000:.1f} mm"
                f" | ridge={max_ridge * 1000:.1f} mm"
                f" | finger_gap={finger_gap * 1000:.1f} mm"
                f" | hold_min_rise={min_rise * 1000:.1f} mm"
                f" | corner_rise={corner_rise * 1000:.1f} mm"
                f" | {'PASS' if passed else 'FAIL'}"
            )

        env.close()


if __name__ == "__main__":
    main()