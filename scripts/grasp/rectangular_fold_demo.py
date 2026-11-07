"""Sequential rectangular cloth fold with one virtual gripper.

Run with the Isaac Lab launcher; see README_rectangular_fold.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from datetime import datetime
import math
import random
import sys
from pathlib import Path

# Isaac Lab imports its recorder while registering tasks, so expose optional
# project-local recording packages before importing Isaac Lab.
if "--no_video" not in sys.argv:
    dependencies = Path(__file__).resolve().parent / "video_dependencies"
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401

from a_single_point_picker import as_tensor
from virtual_attachment_picker import filter_gripper_contacts
from physical_picker import create_picker, physical_grasp_xy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions
from rectangular_fold_geometry import rectangular_pairs, pair_distances
from successful_video import gate_video_recorders
from towel_visual import TowelVisual


class GraspOutcome(Exception):
    """A physical grasp ended the trial early (not a software error)."""

    def __init__(self, outcome, phase):
        super().__init__(f"{outcome} during {phase}")
        self.outcome = outcome
        self.phase = phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_mode", choices=["virtual", "physical"], default="virtual")
    parser.add_argument("--grasp_radius", type=float, default=0.03)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    parser.add_argument("--success_distance", type=float, default=0.02)
    parser.add_argument("--cloth_x", type=float, default=0.50)
    parser.add_argument("--cloth_y", type=float, default=-0.05)
    parser.add_argument("--cloth_yaw_deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cloth_size", type=float, default=0.30)
    parser.add_argument("--mesh_refinement", type=float, default=32.0)
    parser.add_argument("--cloth_thickness", type=float, default=0.003)
    parser.add_argument("--cloth_density", type=float, default=1000.0 / 3)
    parser.add_argument("--table_length", type=float, default=1.5)
    parser.add_argument("--table_width", type=float, default=0.9)
    parser.add_argument("--youngs_modulus", type=float, default=2e5 / 3)
    parser.add_argument("--bend_stiffness", type=float, default=2e4 / 108,
                        help="Soft 3mm towel preset with reduced fold springback")
    parser.add_argument("--no_video", action="store_true", help="Disable automatic MP4 recording")
    # Optional shared cloth/table settings (unset = keep the task defaults, as before)
    parser.add_argument("--poissons_ratio", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None, help="Cloth elasticity and bend damping")
    parser.add_argument("--cloth_friction", type=float, default=None, help="Cloth dynamic friction (static = dynamic + 0.5)")
    parser.add_argument("--table_friction", type=float, default=None, help="Table dynamic friction, min combine (static = dynamic + 0.01)")
    parser.add_argument("--pos_jitter", type=float, default=0.0, help="Max random shift of the cloth center in x and y [m] (per --seed)")
    # Physical grasp (only used with --grasp_mode physical; values from experiment 008)
    parser.add_argument("--open_width", type=float, default=0.04, help="Finger gap before closing [m]")
    parser.add_argument("--pad_half", type=float, default=0.01, help="Half width of a finger pad [m] (estimate)")
    parser.add_argument("--edge_margin", type=float, default=0.005, help="Clearance between a pad and the cloth edge [m]")
    parser.add_argument("--press_depth", type=float, default=0.0, help="TCP target below the cloth surface when closing [m]")
    parser.add_argument("--finger_friction", type=float, default=1.5, help="Finger dynamic friction, max combine (static = dynamic + 0.5)")
    parser.add_argument("--gripper_effort", type=float, default=70.0, help="Franka hand force limit [N] (real hand: 70 N continuous)")
    # Grasp check after closing: lift a little, the corner must follow and stay close to the TCP
    parser.add_argument("--check_lift", type=float, default=0.06, help="Lift used to check the grasp [m] (same height as #14)")
    parser.add_argument("--check_hold_s", type=float, default=1.0, help="Hold time of the grasp check [s]")
    parser.add_argument("--check_min_rise", type=float, default=0.015, help="Corner must rise at least this much [m]")
    parser.add_argument("--quality_max", type=float, default=0.0,
                        help="Corner must stay within this distance of the TCP [m] (#14). 0 = off: "
                             "the grasp only has to lift the corner; success is judged after placing")
    parser.add_argument("--retries", type=int, default=1, help="Re-grasp attempts after a failed grasp check")
    parser.add_argument("--grasp_yaw_deg", type=float, default=90.0,
                        help="Physical grasp only: hand rotation about world z (multiple of 90). "
                             "0 = jaw closes along world y, 90 = along world x (the fold direction)")
    # Transport monitoring
    parser.add_argument("--slip_drift", type=float, default=0.02, help="Corner-to-TCP change counted as slip [m]")
    parser.add_argument("--drop_height", type=float, default=0.01, help="Corner below this height above the table while carried = dropped [m]")
    parser.add_argument("--drop_hand_height", type=float, default=0.06, help="...only counted while the TCP is higher than this above the table [m]")
    parser.add_argument("--stop_after", choices=["probe", "grasp", "transport", "fold"], default="fold",
                        help="probe: settle and report the mesh only, grasp: grasp check, "
                             "transport: first corner placed, fold: whole fold")
    parser.add_argument("--no_gui", action="store_true", help="Run without the Kit window (for sweeps)")
    parser.add_argument("--save_all_videos", action="store_true",
                        help="Write the MP4 for every run, failed ones included (default: successful runs only)")
    parser.add_argument("--video_dir", type=Path, default=Path(__file__).resolve().parents[2] / "videos" / "rectangular_fold")
    add_launcher_args(parser)
    if "--no_gui" not in sys.argv:
        parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    args.video = not args.no_video
    for name in ("lift_height", "settle_seconds", "success_distance", "bend_stiffness",
                 "cloth_size", "mesh_refinement", "table_length", "table_width", "youngs_modulus",
                 "cloth_thickness", "cloth_density"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.mesh_refinement < 1:
        parser.error("mesh_refinement must be at least 1")
    if not all(math.isfinite(v) for v in (args.cloth_x, args.cloth_y, args.cloth_yaw_deg)):
        parser.error("Cloth pose must be finite")
    if args.grasp_mode == "physical" and args.cloth_yaw_deg != 0.0:
        parser.error("Physical grasp assumes an unrotated cloth (the jaw closes along world y)")
    if not math.isfinite(args.grasp_yaw_deg) or args.grasp_yaw_deg % 90 != 0:
        parser.error("grasp_yaw_deg must be a multiple of 90")
    args.hand_yaw_deg = args.grasp_yaw_deg if args.grasp_mode == "physical" else 0.0
    args.jaw_axis = "y" if args.hand_yaw_deg % 180 == 0 else "x"
    rng = random.Random(args.seed)
    args.start_x = args.cloth_x + rng.uniform(-args.pos_jitter, args.pos_jitter)
    args.start_y = args.cloth_y + rng.uniform(-args.pos_jitter, args.pos_jitter)
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    material = cfg.scene.deformable.spawn.physics_material
    original_bend = material.surface_bend_stiffness
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
        material.static_friction = args.cloth_friction + 0.5
        material.dynamic_friction = args.cloth_friction
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    print(f"[fold] cloth={args.cloth_size}m, mesh refinement={args.mesh_refinement}, "
          f"bend stiffness={original_bend:g}->{args.bend_stiffness:g}, Young modulus={args.youngs_modulus:g}", flush=True)
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # update() is called once per env.step, so step at the physics frequency.
    cfg.decimation = 1
    cfg.episode_length_s = 300.0
    cfg.scene.support_neg_y = None
    cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.size = (args.table_length, args.table_width, 1.05)
    cfg.scene.table.spawn.visible = True
    if args.table_friction is not None:
        cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.table_friction + 0.01, dynamic_friction=args.table_friction,
            restitution=0.0, friction_combine_mode="min")
    # The task table is a centered cuboid whose top sits at z = 0
    table_top_z = float(cfg.scene.table.init_state.pos[2]) + 1.05 / 2
    cfg.scene.deformable.init_state.pos = (args.start_x, args.start_y, 0.012)
    half_yaw = math.radians(args.cloth_yaw_deg) / 2
    cfg.scene.deformable.init_state.rot = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cfg.events.reset_deformable = None
    if args.grasp_mode == "virtual":
        # The attachment holds the cloth, so the fingers must not push it.
        cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts, mode="prestartup")
    else:
        # Physical grasp: finger contact and friction hold the cloth (008 gripper settings).
        cfg.scene.robot.spawn.make_uninstanceable = True
        cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.finger_friction + 0.5, dynamic_friction=args.finger_friction,
            restitution=0.0, friction_combine_mode="max")
        cfg.scene.robot.actuators["panda_hand"].joint_effort_limit = args.gripper_effort
        cfg.scene.robot.actuators["panda_hand"].stiffness = 500.0
        cfg.scene.robot.actuators["panda_hand"].damping = 100.0
    # Repeat the same starting arm pose for comparable vibration observations.
    cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
    cfg.curriculum.gravity = None
    cfg.curriculum.action_rate = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    cfg.actions = type(cfg)().actions.ik
    if args.grasp_mode == "physical":
        # One joint moves one finger, so half of the gap
        cfg.actions.gripper_action.open_command_expr = {"panda_finger_joint1": args.open_width / 2}
    # Match the measured TCP used by ee_frame (the stock IK offset is 0.107).
    cfg.actions.arm_action.body_offset.pos = (0.0, 0.0, 0.1034)
    cfg.actions.arm_action.controller.ik_params = {"lambda_val": 0.1}
    for name in vars(cfg.terminations):
        if name != "time_out":
            setattr(cfg.terminations, name, None)
    cfg.viewer.eye = (1.15, 0.60, 0.60)
    cfg.viewer.lookat = (args.cloth_x, args.cloth_y, 0.03)
    cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=cfg.viewer.eye, lookat=cfg.viewer.lookat)

    run_dir = args.video_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    applied = {"cloth_size": float(cfg.scene.deformable.spawn.size[0]), "density": material.density,
               "youngs_modulus": material.youngs_modulus, "bend_stiffness": material.surface_bend_stiffness,
               "cloth_friction": material.dynamic_friction,
               "hand_yaw_deg": args.hand_yaw_deg, "jaw_axis": args.jaw_axis,
               "open_command": cfg.actions.gripper_action.open_command_expr}
    result = {"success": False, "status": "running", "threshold_m": args.success_distance, "applied": applied,
              "grasp_mode": args.grasp_mode, "stop_after": args.stop_after,
              "cloth_start": [args.start_x, args.start_y], "grasps": []}
    transport = {"max_drift_m": 0.0, "slip_phase": None}
    rows = []
    phase = "initialization"
    step_index = 0
    corner_indices = None
    pairs = None
    video_dir = None
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.kit import KitVisualizerCfg

        # Capture at about 30 Hz, preserving simulation time during playback.
        stride = max(1, round(1.0 / (30 * cfg.sim.dt)))
        fps = max(1, round(1.0 / (stride * cfg.sim.dt)))
        video_dir = run_dir
        args.enable_cameras = True
        args.visualizer = ["kit"]
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
            eye=cfg.viewer.eye, lookat=cfg.viewer.lookat,
            window_width=640, window_height=480,
        )]
        cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:kit", output_dir=str(video_dir),
            output_filename_prefix="rectangular_fold", fps=fps, frame_stride=stride,
            video_length=sys.maxsize,
        )]
        # Store metrics alongside each recording so repeat runs remain comparable.
        print(f"[A_single_point] Recording {fps} FPS video to {video_dir}", flush=True)

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        gate_video_recorders(env, lambda: args.save_all_videos
                             or (result["status"] == "completed" and result["success"]))
        picker = None
        try:
            env.reset()
            base = env.unwrapped
            # Success markers are created visible independently of debug_vis.
            # With no success asset they otherwise spawn a table-sized cube at
            # the world origin, hiding the robot and cloth.
            base.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
            scene = base.scene
            cloth, ee = scene["deformable"], scene["ee_frame"]
            towel = TowelVisual(cloth, args.cloth_thickness,
                                Path(__file__).resolve().parents[2] / "assets" / "materials" / "towel")
            result["mesh_node_count"] = int(as_tensor(cloth.data.nodal_pos_w).shape[1])
            print(f"[fold] mesh node count={result['mesh_node_count']}", flush=True)
            robot = scene["robot"]
            result["filtered_gripper_contacts"] = getattr(base, "virtual_gripper_filtered_paths", None)
            finger_ids = robot.find_joints("panda_finger_joint.*")[0]
            if len(finger_ids) != 2:
                raise RuntimeError("Expected two Franka finger joints for gripper verification")
            finger_body_ids = robot.find_bodies("panda_.*finger")[0]
            anchor = lambda: as_tensor(ee.data.target_pos_w)[0, 0]
            # The physical grasp point is about 4 cm from the corner, so its reach check uses the picker default (6 cm)
            picker = (create_picker(cloth, anchor, grasp_mode="physical") if args.grasp_mode == "physical"
                      else create_picker(cloth, anchor, grasp_mode=args.grasp_mode, grasp_radius=args.grasp_radius))
            action = torch.zeros(base.action_space.shape, device=base.device)
            if action.shape != (1, 8):
                raise RuntimeError(f"Expected absolute pose IK action (1, 8), got {action.shape}")
            target_w = anchor().clone()
            target_quat_w = as_tensor(ee.data.target_quat_w)[0, 0].clone()
            # Hand pose used for the approach: identity, rotated about world z (the gripper is 180 deg symmetric)
            half_yaw_hand = math.radians(args.hand_yaw_deg) / 2
            grasp_quat = action.new_tensor((math.cos(half_yaw_hand), 0.0, 0.0, math.sin(half_yaw_hand)))

            def set_pose(position_w, quaternion_w):
                # Absolute IK actions are expressed in the robot ROOT frame,
                # which need not coincide with the environment origin.
                pos_b, quat_b = subtract_frame_transforms(
                    as_tensor(robot.data.root_pos_w), as_tensor(robot.data.root_quat_w),
                    position_w.unsqueeze(0), quaternion_w.unsqueeze(0),
                )
                action[0, :3] = pos_b[0]
                action[0, 3:7] = quat_b[0]

            set_pose(target_w, target_quat_w)
            action[0, 7] = 1.0

            def step():
                nonlocal step_index
                if picker.backend in ("kinematic", "physx_attachment"):
                    picker.update(dt=base.physics_dt)
                towel.update()
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("Episode ended during demonstration; retry with a longer episode")
                if picker.backend == "state_projection":
                    # Enforce the virtual constraint AFTER the cloth solver.
                    # Pre-step-only writes are undone by the stiff cloth solver.
                    picker.update(dt=base.physics_dt)
                step_index += 1
                # Physical grasp: watch the carried corner (corrections use a deliberately low arc)
                if (args.grasp_mode == "physical" and picker.is_grasping and "fold_arc" in phase
                        and not phase.startswith("correction")):
                    drift = picker.drift()
                    transport["max_drift_m"] = max(transport["max_drift_m"], drift)
                    if drift > args.slip_drift and transport["slip_phase"] is None:
                        transport["slip_phase"] = phase
                    # Dropped: the corner lies on the table while the hand is still well above it
                    # (at the end of the arc both come down together, which is not a drop)
                    if (float(picker.target_position_w[2]) - table_top_z < args.drop_height
                            and float(anchor()[2]) - table_top_z > args.drop_hand_height):
                        raise GraspOutcome("dropped", phase)
                if corner_indices is not None and step_index % 4 == 0:
                    rows.append((step_index * base.physics_dt, phase,
                                 action[0].detach().cpu().numpy().copy(),
                                 as_tensor(robot.data.joint_pos)[0].detach().cpu().numpy().copy(),
                                 corners().detach().cpu().numpy().copy(),
                                 anchor().detach().cpu().numpy().copy(), picker.is_grasping))

            def hold(count):
                for _ in range(count):
                    step()

            def set_gripper(closed):
                before = as_tensor(robot.data.joint_pos)[0, finger_ids].clone()
                # BinaryJointPositionAction maps negative to closed, positive to open.
                action[0, 7] = -1.0 if closed else 1.0
                hold(max(60, math.ceil(1.0 / base.physics_dt)))
                after = as_tensor(robot.data.joint_pos)[0, finger_ids].clone()
                state = "closed" if closed else "open"
                print(f"[virtual_grasp] gripper {state}: joints={after.tolist()}, gap={float(after.sum()):.4f} m", flush=True)
                # Check physical joint motion, rather than just the command sign.
                if closed and float(before.sum() - after.sum()) < 0.005:
                    raise RuntimeError("Gripper close command did not reduce the measured finger gap")
                if not closed and float(after.sum() - before.sum()) < 0.005:
                    raise RuntimeError("Gripper open command did not increase the measured finger gap")

            def corners():
                return get_corner_world_positions(as_tensor(cloth.data.nodal_pos_w)[0], corner_indices)

            def move(goal_w, count=240, goal_quat_w=None, strict=True):
                nonlocal target_w, target_quat_w
                start = target_w.clone()
                start_quat = target_quat_w.clone()
                goal_quat = target_quat_w.clone() if goal_quat_w is None else goal_quat_w.clone()
                if float(torch.dot(start_quat, goal_quat)) < 0.0:
                    goal_quat = -goal_quat
                for i in range(1, count + 1):
                    t = i / count
                    alpha = t * t * (3 - 2 * t)
                    quat = torch.lerp(start_quat, goal_quat, alpha)
                    quat = quat / torch.linalg.vector_norm(quat)
                    set_pose(torch.lerp(start, goal_w, alpha), quat)
                    step()
                target_w = goal_w.clone()
                target_quat_w = goal_quat
                # Fixed-duration interpolation alone does not guarantee arrival.
                for _ in range(1200 if strict else 60):
                    step()
                    if float(torch.linalg.vector_norm(anchor() - goal_w)) < 0.005:
                        break
                error = float(torch.linalg.vector_norm(anchor() - goal_w))
                print(f"[virtual_grasp] move goal={goal_w.tolist()}, actual={anchor().tolist()}, error={error:.4f} m", flush=True)
                if strict and error >= 0.005:
                    raise RuntimeError(f"IK did not reach target: {error:.4f} m; goal={goal_w.tolist()}, actual={anchor().tolist()}")

            physical = args.grasp_mode == "physical"

            def grasp_target(source):
                """Where the TCP goes down: the corner itself (virtual) or the 008 point near it (physical)."""
                corner = corners()[source].clone()
                if physical:
                    center_xy = corners()[:, :2].mean(dim=0)
                    corner[:2] = physical_grasp_xy(corner[:2], center_xy, args.open_width,
                                                   pad_half=args.pad_half, edge_margin=args.edge_margin,
                                                   jaw_axis=args.jaw_axis)
                return corner

            def grasp_check(source, number):
                """Lift a little and hold: the corner must follow and stay close to the TCP."""
                nonlocal phase
                phase = f"{number}_grasp_check"
                start_z = float(corners()[source][2])
                lift = anchor().clone()
                lift[2] += args.check_lift
                move(lift, 60, strict=False)
                hold(math.ceil(args.check_hold_s / base.physics_dt))
                corner = corners()[source]
                rise = float(corner[2]) - start_z
                corner_tcp = float(torch.linalg.vector_norm(corner - anchor()))
                ok = rise >= args.check_min_rise and (args.quality_max <= 0 or corner_tcp <= args.quality_max)
                return {"rise_m": rise, "corner_tcp_m": corner_tcp, "ok": ok}

            def place_corner(source, destination, number, correction=False):
                nonlocal phase
                vertex = int(corner_indices[source])
                record = {"number": str(number), "source": int(source), "attempts": 0, "ok": False}
                result["grasps"].append(record)
                for attempt in range(1 + (args.retries if physical else 0)):
                    record["attempts"] = attempt + 1
                    phase = f"{number}_approach"
                    above = grasp_target(source)
                    above[2] += 0.08
                    # Raise vertically before moving across the cloth.
                    safe = anchor().clone()
                    safe[2] = max(float(safe[2]), float(above[2]))
                    move(safe, 120)
                    move(above, goal_quat_w=grasp_quat)
                    # Gap in free air, to tell "never opened fully" from "squeezed while descending"
                    record["open_gap_above_m"] = float(as_tensor(robot.data.joint_pos)[0, finger_ids].sum())
                    if len(finger_body_ids) == 2:
                        # World vector between the two fingers: its larger component is the real jaw axis
                        fingers = as_tensor(robot.data.body_pos_w)[0, finger_body_ids]
                        record["finger_vec_open_mm"] = [round(float(v) * 1000, 1) for v in fingers[0] - fingers[1]]
                        print(f"[fold] finger vector (open, world) mm={record['finger_vec_open_mm']}, "
                              f"expected jaw axis={args.jaw_axis}", flush=True)
                    phase = f"{number}_descend"
                    touch = grasp_target(source)
                    # Virtual: stop 1.5 cm above the corner. Physical: fingertips on the cloth (blocked by the table)
                    touch[2] += -args.press_depth if physical else 0.015
                    move(touch, 180, strict=not physical)
                    phase = f"{number}_grasp"
                    record["open_gap_m"] = float(as_tensor(robot.data.joint_pos)[0, finger_ids].sum())
                    set_gripper(closed=True)
                    current = corners()[source].clone()
                    if not picker.grasp(current, vertex_id=vertex):
                        if not physical:
                            raise RuntimeError(f"Corner {source} is outside grasp radius")
                        record.update(ok=False, reason="gripper_not_near_corner")
                    elif not physical:
                        record.update(ok=True)
                    else:
                        record.update(grasp_check(source, number))
                        if record["ok"]:
                            # Measure slip from the settled grip, not from the moment of closing
                            picker.grasp_offset = picker.corner_offset().clone()
                    print(f"[fold] grasp {number} attempt {attempt + 1}: {record}", flush=True)
                    if record["ok"]:
                        break
                    picker.release()
                    set_gripper(closed=False)
                    retreat = anchor().clone()
                    retreat[2] += 0.05
                    move(retreat, 120, strict=False)
                else:
                    raise GraspOutcome("grasp_failed", phase)
                if args.stop_after == "grasp":
                    phase = f"{number}_release"
                    set_gripper(closed=False)
                    picker.release()
                    return
                # Sweep around the fold line. A high horizontal translation
                # pulled the far edge away and exceeded the robot workspace.
                offset = picker.target_position_w - anchor()
                start_node = corners()[source].clone()
                initial_span = float(torch.linalg.vector_norm(corners()[destination, :2] - start_node[:2]))
                arc_height = (max(0.008, min(args.lift_height, initial_span / 2)) if correction
                                  else max(args.lift_height, initial_span / 2))
                for segment in range(1, 9):
                    phase = f"{number}_fold_arc_{segment}"
                    if physical:
                        offset = picker.target_position_w - anchor()
                    t = segment / 8
                    alpha = (1 - math.cos(math.pi * t)) / 2
                    destination_now = corners()[destination].clone()
                    node_goal = torch.lerp(start_node, destination_now, alpha)
                    node_goal[2] += arc_height * math.sin(math.pi * t) + max(0.003, args.cloth_thickness + 0.001) * alpha
                    move(node_goal - offset, 60 if correction else 120)
                phase = f"{number}_lower"
                for _ in range(3):
                    if physical:
                        offset = picker.target_position_w - anchor()
                    goal = corners()[destination].clone() - offset
                    goal[2] += max(0.003, args.cloth_thickness + 0.001)
                    # Physical: fingers collide with the cloth layers under them, so the TCP cannot
                    # reach this height. Go as low as possible; success is judged by the corner
                    # distance after release, not by the TCP height.
                    move(goal, 120, strict=not physical)
                    if physical:
                        record["lower_error_m"] = float(torch.linalg.vector_norm(anchor() - goal))
                phase = f"{number}_release"
                record["max_drift_m"] = transport["max_drift_m"]
                set_gripper(closed=False)
                picker.release()
                phase = f"{number}_retreat"
                retreat = anchor().clone()
                retreat[2] += 0.10
                move(retreat, 180)
                hold(math.ceil(args.settle_seconds / base.physics_dt))
                print(f"[fold] after placement {number}: distances={pair_distances(corners(), pairs)}", flush=True)


            with torch.inference_mode():
                hold(120)
                corner_indices = detect_corner_vertex_indices(as_tensor(cloth.data.nodal_pos_w)[0])
                pairs = rectangular_pairs(corners(), as_tensor(robot.data.root_pos_w)[0, :2])
                result["corner_vertex_ids"] = corner_indices.tolist()
                result["pairs"] = pairs
                result["initial_distances_m"] = pair_distances(corners(), pairs)
                print(f"[fold] tracked vertices={corner_indices.tolist()}, pairs={pairs}", flush=True)
                if args.stop_after == "probe":
                    result.update(status="completed", success=True)
                    return
                for number, (source, destination) in enumerate(pairs, 1):
                    place_corner(source, destination, number)
                    if args.stop_after != "fold":
                        break
                if args.stop_after != "fold":
                    # grasp: the grasp check passed. transport: the first corner reached its goal.
                    phase = "evaluation"
                    hold(math.ceil(args.settle_seconds / base.physics_dt))
                    distances = pair_distances(corners(), pairs)
                    first_ok = math.isfinite(distances[0]) and distances[0] <= args.success_distance
                    result.update(status="completed", final_distances_m=distances,
                                  success=all(g["ok"] for g in result["grasps"])
                                  and (args.stop_after == "grasp" or first_ok))
                    print(f"[fold] {args.stop_after} check success={result['success']}, distances={distances}", flush=True)
                    return
                result["correction_count"] = 0
                # Re-read the released cloth and correct only the worst pair.
                # Tiny corrections use a low arc rather than lifting 12cm again.
                for correction in range(2):
                    phase = "correction_evaluation"
                    hold(math.ceil(args.settle_seconds / base.physics_dt))
                    distances = pair_distances(corners(), pairs)
                    if all(d <= args.success_distance for d in distances):
                        break
                    worst = max(range(len(pairs)), key=distances.__getitem__)
                    print(f"[fold] correction {correction + 1}: pair={worst}, error={distances[worst]:.4f} m", flush=True)
                    place_corner(*pairs[worst], number=f"correction_{correction + 1}", correction=True)
                    result["correction_count"] += 1
                phase = "evaluation"
                hold(math.ceil(args.settle_seconds / base.physics_dt))
                distances = pair_distances(corners(), pairs)
                result.update(status="completed", final_distances_m=distances,
                              final_corner_positions_w=corners().tolist(),
                              final_rms_speed_m_s=float(torch.sqrt((as_tensor(cloth.data.nodal_vel_w)[0] ** 2).sum(-1).mean())),
                              success=all(math.isfinite(d) and d <= args.success_distance for d in distances))
                print(f"[fold] success={result['success']}, final distances={distances}", flush=True)
        except GraspOutcome as outcome:
            result.update(status=outcome.outcome, phase=outcome.phase, success=False)
            print(f"[fold] {outcome.outcome} during {outcome.phase}", flush=True)
        except BaseException as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", phase=phase)
            raise
        finally:
            if picker is not None:
                picker.reset()
            try:
                env.close()
            finally:
                (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                nodes = result.get("mesh_node_count") or 0
                summary = {
                    "status": result["status"], "success": result["success"], "phase": result.get("phase"),
                    "grasp_mode": args.grasp_mode, "stop_after": args.stop_after, "seed": args.seed,
                    "cloth_start": result["cloth_start"], "mesh_node_count": nodes,
                    "spacing_mm": args.cloth_size * 1000 / max(math.sqrt(nodes) - 1, 1) if nodes else None,
                    "grasps": result["grasps"], "max_drift_m": transport["max_drift_m"],
                    "slip_phase": transport["slip_phase"], "final_distances_m": result.get("final_distances_m"),
                    "applied": applied, "run_dir": str(run_dir),
                }
                print("[fold] RESULT " + json.dumps(summary), flush=True)
                if rows:
                    # Keep every trial's diagnostic measurements. Only successful
                    # trials are exported to the imitation-learning dataset.
                    with (run_dir / "steps.csv").open("w", newline="", encoding="utf-8") as output:
                        writer = csv.writer(output)
                        writer.writerow(["time_s", "phase", "grasping", "pair_1_distance_m", "pair_2_distance_m"])
                        for t, label, _, _, points, _, grasping in rows:
                            distances = pair_distances(torch.from_numpy(points), pairs)
                            writer.writerow([t, label, grasping, *distances])
                    if result["success"]:
                        np.savez_compressed(run_dir / "demonstration.npz",
                                            time_s=np.array([r[0] for r in rows]),
                                            phase=np.array([r[1] for r in rows]),
                                            action=np.stack([r[2] for r in rows]),
                                            joint_pos=np.stack([r[3] for r in rows]),
                                            corners_w=np.stack([r[4] for r in rows]),
                                            tcp_w=np.stack([r[5] for r in rows]),
                                            grasping=np.array([r[6] for r in rows]))
            if video_dir is not None and result["success"]:
                clips = list(video_dir.glob("*.mp4"))
                if not clips or not all(clip.stat().st_size > 0 for clip in clips):
                    raise RuntimeError(f"No video was saved; check the recording errors above: {video_dir}")
                for clip in clips:
                    print(f"[A_single_point] Video saved: {clip}", flush=True)
            elif video_dir is not None:
                print("[fold] Unsuccessful trial: video frames discarded without saving MP4.", flush=True)


if __name__ == "__main__":
    main()
