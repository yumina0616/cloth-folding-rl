"""
Physical cloth grasp by pressing and closing (experiment 006).
The fingertips press the cloth near a corner and close,
so that friction gathers the cloth into a ridge between the fingers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
# [007] Open finger gap. The grasp point inset is computed from it so both fingertips land on the cloth
parser.add_argument("--open_width", type=float, default=0.08, help="Finger gap when the gripper is open [m] (max 0.08)")
parser.add_argument("--pad_half", type=float, default=0.01, help="Half width of a finger pad [m] (approximate)")
parser.add_argument("--edge_margin", type=float, default=0.005, help="Clearance between a finger pad and the cloth edge [m]")
# [007] Cloth properties
parser.add_argument("--thickness", type=float, default=0.006, help="Cloth surface thickness [m]")
parser.add_argument("--bend_scale", type=float, default=1.0, help="Multiplier on the default surface bend stiffness")
parser.add_argument("--edge_refinement", type=int, default=8, help="Cloth mesh subdivisions per edge (8 -> 9x9 vertices)")
parser.add_argument("--no_gui", action="store_true", help="Run without the Kit window (faster, for sweeps)")
# [008] Table friction and repeated trials
parser.add_argument("--table_friction", type=float, default=0.03, help="Table dynamic friction (static = dynamic + 0.01)")
parser.add_argument("--seed", type=int, default=0, help="Seed for the cloth position jitter")
parser.add_argument("--pos_jitter", type=float, default=0.0, help="Max random shift of the cloth start position in x and y [m]")
# [008] Where to put the grasp point: "axis" moves it closer to the corner, "diagonal" is the 007 layout
parser.add_argument("--grasp_layout", choices=("axis", "diagonal"), default="axis")
parser.add_argument("--linger", type=float, default=0.0, help="Keep simulating (holding the cloth) this many seconds after the result, so the window stays open for recording")
parser.add_argument("--quality_max", type=float, default=0.03, help="Grasp quality: max corner-to-gripper distance while held [m] (#14)")
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=220)
parser.add_argument("--lift_height", type=float, default=0.060)
parser.add_argument("--lift_success_height", type=float, default=0.020)
parser.add_argument("--hold_steps", type=int, default=180)

add_launcher_args(parser)
# Without a visualizer the app runs headless
if "--no_gui" not in sys.argv:
    parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

# [008] Same seed -> same cloth shift, so a run can be reproduced
TABLE_TOP_Z = 0.0
_rng = random.Random(args_cli.seed)
cloth_offset = (_rng.uniform(-args_cli.pos_jitter, args_cli.pos_jitter),
                _rng.uniform(-args_cli.pos_jitter, args_cli.pos_jitter))

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
    # [008] Table top height, to see how far above it the cloth comes to rest (assumes a centered cuboid)
    global TABLE_TOP_Z
    TABLE_TOP_Z = float(env_cfg.scene.table.init_state.pos[2]) + env_cfg.scene.table.spawn.size[2] / 2
    env_cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=args_cli.table_friction + 0.01,
        dynamic_friction=args_cli.table_friction,
        restitution=0.0,
        friction_combine_mode="min",
    )

    # [006] Place the cloth at the table center (no table edge needed)
    env_cfg.scene.deformable.init_state.pos = (0.5 + cloth_offset[0], 0.0 + cloth_offset[1], 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cloth_material = env_cfg.scene.deformable.spawn.physics_material
    cloth_material.surface_thickness = args_cli.thickness
    cloth_material.density = 150.0
    cloth_material.static_friction = 2.0
    cloth_material.elasticity_damping = 0.25
    cloth_material.dynamic_friction = 1.5
    cloth_material.bend_damping = 0.25
    # [007] Scale the task default; fails loudly if a non-PhysX cloth config is selected
    cloth_material.surface_bend_stiffness *= args_cli.bend_scale
    env_cfg.scene.deformable.spawn.edge_refinement = args_cli.edge_refinement

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
    # [007] Open finger joint target. One joint moves one finger, so half of the gap
    env_cfg.actions.gripper_action.open_command_expr = {"panda_finger_joint1": args_cli.open_width / 2}

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
            corner_index = int(corner_indices[slot].item())
            # [008] Resting height of the cloth above the table top (collision thickness check)
            settle_gap = float(nodes_w[:, 2].median().item()) - TABLE_TOP_Z

            center_xy = corners_w[:, :2].mean(dim=0)
            inward = center_xy - corner_w[:2]
            inward = inward / torch.linalg.vector_norm(inward)
            # Both fingertips must land on the cloth. The jaw closes along y, so the fingers
            # spread only in y: y needs room for half the gap, x only for the pad itself
            pad_room = args_cli.pad_half + args_cli.edge_margin
            jaw_room = args_cli.open_width / 2 + pad_room
            if args_cli.grasp_layout == "axis":
                # [008] x: pad only, y: half gap + pad -> closer to the corner
                toward_center = torch.sign(center_xy - corner_w[:2])
                grasp_xy = corner_w[:2] + toward_center * corner_w.new_tensor((pad_room, jaw_room))
            else:
                # [007] Same distance from both edges, on the 45 deg diagonal
                grasp_xy = corner_w[:2] + math.sqrt(2.0) * jaw_room * inward
            inset = float(torch.linalg.vector_norm(grasp_xy - corner_w[:2]).item())
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

            # [007] Check where the open fingers are. Finger body origins, not the exact pad tips
            open_gap = float(robot.data.joint_pos.torch[0, finger_joint_ids].sum().item())
            cloth_min = corners_w[:, :2].min(dim=0).values
            cloth_max = corners_w[:, :2].max(dim=0).values

            def inside_dist(point_w: torch.Tensor) -> float:
                """Distance from a point to the nearest cloth edge in xy (negative = outside)."""
                gaps = torch.cat((point_w[:2] - cloth_min, cloth_max - point_w[:2]))
                return float(gaps.min().item())

            left_in = inside_dist(left_w)
            right_in = inside_dist(right_w)
            left_z = float(left_w[2].item()) - node_z
            right_z = float(right_w[2].item()) - node_z
            print(
                f"[press_close] open_gap={open_gap * 1000:.1f} mm (requested {args_cli.open_width * 1000:.1f})"
                f" | inset={inset * 1000:.1f} mm"
                f" | finger inside cloth: left={left_in * 1000:.1f} mm, right={right_in * 1000:.1f} mm"
                f" | finger origin above cloth: left={left_z * 1000:.1f} mm, right={right_z * 1000:.1f} mm"
            )

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
            corner_tcp_max = 0.0
            for _ in range(args_cli.hold_steps):
                hold(1, -1.0)
                rise = float(cloth.data.nodal_pos_w.torch[0, grasp_node, 2].item()) - cloth_z
                min_rise = min(min_rise, rise)
                # [008] How far the corner hangs from the gripper center (TCP)
                tcp_w = ee_frame.data.target_pos_w.torch[0, 0]
                corner_now = cloth.data.nodal_pos_w.torch[0, corner_index]
                corner_tcp = float(torch.linalg.vector_norm(corner_now - tcp_w).item())
                corner_tcp_max = max(corner_tcp_max, corner_tcp)
            gathered = open_gap - finger_gap  # [008] cloth width pulled in between the fingers
            quality_ok = corner_tcp <= args_cli.quality_max
            corner_rise = float(cloth.data.nodal_pos_w.torch[0, corner_index, 2].item() - corner_w[2].item())

            passed = min_rise >= args_cli.lift_success_height
            print(
                f"[press_close] SUMMARY | depth={args_cli.press_depth * 1000:.1f} mm"
                f" | press_gap={press_gap * 1000:.1f} mm"
                f" | ridge={max_ridge * 1000:.1f} mm"
                f" | finger_gap={finger_gap * 1000:.1f} mm"
                f" | hold_min_rise={min_rise * 1000:.1f} mm"
                f" | corner_rise={corner_rise * 1000:.1f} mm"
                f" | corner_to_gripper={corner_tcp * 1000:.1f} mm"
                f" | gathered={gathered * 1000:.1f} mm"
                f" | {'PASS' if passed else 'FAIL'}"
                f" | quality {'OK' if quality_ok else 'NG'}"
            )

            # [007] One machine-readable line for the sweep runner (lengths in mm)
            result = {
                "press_depth": args_cli.press_depth * 1000,
                "table_friction": args_cli.table_friction,
                "grasp_layout": args_cli.grasp_layout,
                "seed": args_cli.seed,
                "cloth_dx": cloth_offset[0] * 1000,
                "cloth_dy": cloth_offset[1] * 1000,
                "settle_gap": settle_gap * 1000,
                "open_width": args_cli.open_width * 1000,
                "thickness": args_cli.thickness * 1000,
                "bend_scale": args_cli.bend_scale,
                "edge_refinement": args_cli.edge_refinement,
                "num_nodes": int(nodes_w.shape[0]),
                "inset": inset * 1000,
                "open_gap": open_gap * 1000,
                "left_in": left_in * 1000,
                "right_in": right_in * 1000,
                "left_z": left_z * 1000,
                "right_z": right_z * 1000,
                "press_gap": press_gap * 1000,
                "ridge": max_ridge * 1000,
                "finger_gap": finger_gap * 1000,
                "hold_min_rise": min_rise * 1000,
                "corner_rise": corner_rise * 1000,
                "corner_tcp": corner_tcp * 1000,
                "corner_tcp_max": corner_tcp_max * 1000,
                "gathered": gathered * 1000,
                "quality_ok": bool(quality_ok),
                "passed": bool(passed),
            }
            print("[press_close] RESULT " + json.dumps(result))

            # Keep the window alive for screen recording (env step = 1/30 s)
            if args_cli.linger > 0:
                print(f"[press_close] keeping the window open for {args_cli.linger:.0f} s")
                hold(int(args_cli.linger * 30), -1.0)

        env.close()


if __name__ == "__main__":
    main()