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
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, subtract_frame_transforms
from isaaclab_tasks.utils import setup_preset_cli
from task_config_compat import resolve_cloth_task_config
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401

from a_single_point_picker import as_tensor
from virtual_attachment_picker import filter_gripper_contacts
from physical_picker import create_picker, pinch_vertex
from finger_pad_probe import probe_finger_pads
from run_config import DEFAULT_CONFIG, apply_config_defaults
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
# Project root, for the shared dataset recorder in scripts/learning (same as main's fold demo)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fold_dataset_export import FoldEpisodeLog, fold_context
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions
from rectangular_fold_geometry import (axis_pairs, cloth_yaw_from_corners, rectangular_pairs, pair_distances,
                                       edge_reference_vertices, layer_jaw_yaw, rotate_nodes_about_center,
                                       to_cloth_frame)
from scripts.metrics.fold_shape_metric import fold_shape_success
from successful_video import gate_video_recorders
from towel_visual import TowelVisual


class GraspOutcome(Exception):
    """A physical grasp ended the trial early (not a software error)."""

    def __init__(self, outcome, phase):
        super().__init__(f"{outcome} during {phase}")
        self.outcome = outcome
        self.phase = phase


class CorrectionNotNeeded(Exception):
    """The released cloth settled into a successful fold while a correction was approaching."""

    def __init__(self, phase):
        super().__init__(phase)
        self.phase = phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_mode", choices=["virtual", "physical"], default="virtual")
    parser.add_argument("--grasp_radius", type=float, default=0.03)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--settle_seconds", type=float, default=3.0)
    parser.add_argument("--success_distance", type=float, default=0.02)
    parser.add_argument(
        "--shape_iou_threshold",
        type=float,
        default=0.75,
        help="Minimum IoU for successful fold shape",
    )
    parser.add_argument("--cloth_x", type=float, default=0.50)
    parser.add_argument("--cloth_y", type=float, default=-0.05)
    parser.add_argument("--cloth_yaw_deg", type=float, default=0.0, help="Cloth rotation about world z [deg]")
    parser.add_argument("--yaw_jitter", type=float, default=0.0,
                        help="Max random change of --cloth_yaw_deg [deg] (per --seed, drawn after --pos_jitter)")
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
    parser.add_argument("--cloth_friction", type=float, default=None, help="Cloth dynamic friction (None = task default)")
    parser.add_argument("--cloth_static_friction", type=float, default=None,
                        help="Cloth static friction (None = same as --cloth_friction)")
    parser.add_argument("--table_friction", type=float, default=None,
                        help="Table dynamic friction, min combine (None = task default material)")
    parser.add_argument("--table_static_friction", type=float, default=None,
                        help="Table static friction (None = same as --table_friction)")
    parser.add_argument("--pos_jitter", type=float, default=0.0, help="Max random shift of the cloth center in x and y [m] (per --seed)")
    # Physical grasp (only used with --grasp_mode physical; values from experiment 008)
    parser.add_argument("--open_width", type=float, default=0.04, help="Finger gap before closing [m]")
    parser.add_argument("--pad_half", type=float, default=0.01, help="Half width of a finger pad [m] (estimate)")
    parser.add_argument("--edge_margin", type=float, default=0.005, help="Clearance between a pad and the cloth edge [m]")
    parser.add_argument("--pinch_jaw_shift", type=float, default=0.0,
                        help="Physical: move the pinch point this much toward the corner along the jaw axis only "
                             "(world y at grasp_yaw_deg 0) [m]. 0 = both open fingertips land on the cloth")
    parser.add_argument("--close_lift", type=float, default=0.0,
                        help="Physical: raise the TCP this much above where the descent stopped before closing [m]")
    parser.add_argument("--arc_peak", type=float, default=None,
                        help="Carry along a parabola from the grasp-check height; peak = this much above the corner start [m]")
    parser.add_argument("--min_tcp_height", type=float, default=0.005,
                        help="Physical carry arc: never aim the TCP lower than this above the table [m]")
    parser.add_argument("--release_height", type=float, default=None,
                        help="Physical: open the fingers with the TCP this high above the table [m]")
    parser.add_argument("--move_axis", choices=["auto", "x", "y"], default="auto",
                        help="auto = edge nearest the robot (old); x/y = move corners along this world axis")
    parser.add_argument("--place_inset", type=float, default=0.0,
                        help="Physical mirror place: pull the place point back toward the fold line by this much [m]")
    parser.add_argument("--free_speed", type=float, default=1.0,
                        help="Speed-up factor for moves while no cloth is held (2 = twice as fast)")
    parser.add_argument("--retry_in_place", action="store_true",
                        help="After a failed grasp check, open and re-grasp from where the hand is (no raise)")
    parser.add_argument("--approach_order", choices=["direct", "yx"], default="direct",
                        help="Free move to above the grasp point: direct = one straight line (old), yx = world y first, then x")
    parser.add_argument("--place_settle_s", type=float, default=None,
                        help="Wait after each placement before the next corner [s] (None = --settle_seconds)")
    parser.add_argument("--press_depth", type=float, default=0.0, help="TCP target below the cloth surface when closing [m]")
    parser.add_argument("--finger_friction", type=float, default=1.5, help="Finger dynamic friction, max combine")
    parser.add_argument("--finger_static_friction", type=float, default=2.0, help="Finger static friction, max combine")
    parser.add_argument("--gripper_effort", type=float, default=70.0, help="Franka hand force limit [N] (real hand: 70 N continuous)")
    # Grasp check after closing: lift a little, the corner must follow and stay close to the TCP
    parser.add_argument("--check_lift", type=float, default=0.06, help="Lift used to check the grasp [m] (same height as #14)")
    parser.add_argument("--check_hold_s", type=float, default=1.0, help="Hold time of the grasp check [s]")
    parser.add_argument("--grasp_settle_s", type=float, default=0.5,
                        help="Physical grasp: extra wait after the gripper has closed, before lifting [s] "
                             "(0 = old behaviour, lift right after the 1 s close)")
    parser.add_argument("--check_lift_s", type=float, default=2.0,
                        help="Physical grasp: duration of the grasp-check lift [s] (0.5 = old behaviour, 60 steps)")
    parser.add_argument("--check_min_rise", type=float, default=0.015, help="Corner must rise at least this much [m]")
    parser.add_argument("--quality_max", type=float, default=0.0,
                        help="Corner must stay within this distance of the TCP [m] (#14). 0 = off: "
                             "the grasp only has to lift the corner; success is judged after placing")
    parser.add_argument("--retries", type=int, default=1, help="Re-grasp attempts after a failed grasp check")
    parser.add_argument("--stack_check_lift", action="store_true",
                        help="Old behaviour: build the carry arc on top of the height the grasp check lifted to")
    parser.add_argument("--place_mode", choices=["mirror", "live"], default="mirror",
                        help="Physical grasp: where the hand goes when placing. mirror = the point where the destination "
                             "corner would be pinched (the held corner lands on it); live = destination minus the "
                             "measured corner-to-hand offset (old behaviour, kept for comparison)")
    parser.add_argument("--grasp_yaw_deg", type=float, default=0.0,
                        help="Physical grasp only: hand rotation relative to the cloth (multiple of 90). "
                             "0 (default, original) = jaw closes along the cloth y axis, 90 = along the cloth x axis. "
                             "The measured cloth rotation is added on top")
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
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="JSON of cloth/table/finger settings used as defaults (009 unified cloth); "
                             "command-line options still override it. 'none' = the script defaults only")
    apply_config_defaults(parser, sys.argv[1:])
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
    if not math.isfinite(args.shape_iou_threshold) or not 0 <= args.shape_iou_threshold <= 1:
        parser.error("--shape_iou_threshold must be between 0 and 1")
    if args.mesh_refinement < 1:
        parser.error("mesh_refinement must be at least 1")
    if not all(math.isfinite(v) for v in (args.cloth_x, args.cloth_y, args.cloth_yaw_deg)):
        parser.error("Cloth pose must be finite")
    if not (math.isfinite(args.yaw_jitter) and 0 <= args.yaw_jitter < 45):
        parser.error("yaw_jitter must be in [0, 45) deg")
    if not math.isfinite(args.grasp_yaw_deg) or args.grasp_yaw_deg % 90 != 0:
        parser.error("grasp_yaw_deg must be a multiple of 90")
    if not (math.isfinite(args.grasp_settle_s) and args.grasp_settle_s >= 0):
        parser.error("grasp_settle_s must be >= 0")
    if not (math.isfinite(args.check_lift_s) and args.check_lift_s > 0):
        parser.error("check_lift_s must be > 0")
    # Static and dynamic friction are set separately; a missing static value equals the dynamic one
    for name in ("cloth", "table", "finger"):
        dynamic = getattr(args, f"{name}_friction")
        static = getattr(args, f"{name}_static_friction")
        if static is not None and dynamic is None:
            parser.error(f"--{name}_static_friction needs --{name}_friction")
        if dynamic is None:
            continue
        static = dynamic if static is None else static
        if not (math.isfinite(dynamic) and math.isfinite(static) and 0 <= dynamic <= static):
            parser.error(f"{name} friction must satisfy 0 <= dynamic ({dynamic}) <= static ({static})")
        setattr(args, f"{name}_static_friction", static)
    args.hand_yaw_deg = args.grasp_yaw_deg if args.grasp_mode == "physical" else 0.0
    args.jaw_axis = "y" if args.hand_yaw_deg % 180 == 0 else "x"
    rng = random.Random(args.seed)
    args.start_x = args.cloth_x + rng.uniform(-args.pos_jitter, args.pos_jitter)
    args.start_y = args.cloth_y + rng.uniform(-args.pos_jitter, args.pos_jitter)
    # Drawn after the position, so earlier seeds keep the same position
    args.start_yaw_deg = args.cloth_yaw_deg + (rng.uniform(-args.yaw_jitter, args.yaw_jitter) if args.yaw_jitter else 0.0)
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, _ = resolve_cloth_task_config("Isaac-Lift-Cloth-Franka", "")
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
        material.static_friction = args.cloth_static_friction
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
            static_friction=args.table_static_friction, dynamic_friction=args.table_friction,
            restitution=0.0, friction_combine_mode="min")
    # The task table is a centered cuboid whose top sits at z = 0
    table_top_z = float(cfg.scene.table.init_state.pos[2]) + 1.05 / 2
    cfg.scene.deformable.init_state.pos = (args.start_x, args.start_y, 0.012)
    # Spawn unrotated: a rotated spawn left the cloth crumpled (commanded 40 deg, corner pairs 0.23-0.27 m
    # instead of 0.30, measured yaw ~0). The settled flat cloth is rotated later by writing its nodes.
    cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.events.reset_deformable = None
    if args.grasp_mode == "virtual":
        # The attachment holds the cloth, so the fingers must not push it.
        cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts, mode="prestartup")
    else:
        # Physical grasp: finger contact and friction hold the cloth (008 gripper settings).
        cfg.scene.robot.spawn.make_uninstanceable = True
        cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.finger_static_friction, dynamic_friction=args.finger_friction,
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
               "cloth_friction": material.dynamic_friction, "cloth_static_friction": material.static_friction,
               "table_friction": args.table_friction, "table_static_friction": args.table_static_friction,
               "finger_friction": args.finger_friction if args.grasp_mode == "physical" else None,
               "finger_static_friction": args.finger_static_friction if args.grasp_mode == "physical" else None,
               "stack_check_lift": args.stack_check_lift,
               "hand_yaw_deg": args.hand_yaw_deg, "jaw_axis": args.jaw_axis, "place_mode": args.place_mode,
               "grasp_settle_s": args.grasp_settle_s, "pinch_jaw_shift": args.pinch_jaw_shift, "check_lift_s": args.check_lift_s,
               "open_command": cfg.actions.gripper_action.open_command_expr}
    result = {"success": False, "status": "running", "threshold_m": args.success_distance, "applied": applied,
              "grasp_mode": args.grasp_mode, "stop_after": args.stop_after,
              "cloth_start": [args.start_x, args.start_y], "cloth_start_yaw_deg": args.start_yaw_deg, "grasps": []}
    transport = {"max_drift_m": 0.0, "slip_phase": None}
    rows = []
    episode_log = FoldEpisodeLog()  # every step, saved in the PR #37 BC format for successful runs
    phase = "initialization"
    step_index = 0
    corner_indices = None
    pairs = None
    initial_center_xy = None  # centre of the four corners while the cloth is still flat
    pinch_vertices = None     # per corner: the cloth vertex to pinch (physical grasp), chosen on the flat cloth
    edge_vertices = None      # per corner: (vertex along the x edge, vertex along the y edge) on the flat cloth
    video_dir = None
    if args.video:
        # Only the launch options here: Isaac Lab 3.0 refuses isaaclab_visualizers imports
        # before SimulationApp exists, so the recorder config is built after launch (below).
        video_dir = run_dir
        args.enable_cameras = True
        args.visualizer = ["kit"]

    with launch_simulation(cfg, args):
        if args.video:
            from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
            from isaaclab_visualizers.kit import KitVisualizerCfg

            # Capture at about 30 Hz, preserving simulation time during playback.
            stride = max(1, round(1.0 / (30 * cfg.sim.dt)))
            fps = max(1, round(1.0 / (stride * cfg.sim.dt)))
            cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
                eye=cfg.viewer.eye, lookat=cfg.viewer.lookat,
                window_width=640, window_height=480,
            )]
            cfg.video_recorders = [VideoRecorderCfg(
                source="visualizer:kit", output_dir=str(video_dir),
                output_filename_prefix="rectangular_fold", fps=fps, frame_stride=stride,
                video_length=sys.maxsize,
            )]
            print(f"[A_single_point] Recording {fps} FPS video to {video_dir}", flush=True)
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
            # Isaac Lab v3.0.0 (release) does not create the marker at all, so it can be None.
            success_marker = getattr(base.command_manager.get_term("deformable_pose"), "success_visualizer", None)
            if success_marker is not None:
                success_marker.set_visibility(False)
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

            def yaw_grasp_quat(yaw_deg):
                # Top-down hand pose used so far, (1, 0, 0, 0), turned about WORLD z by yaw_deg.
                # Built with Isaac Lab's own quaternion functions: a hand-written (cos, 0, 0, sin) tilted
                # the hand instead of turning it, so the tuple order is not assumed here.
                down = action.new_tensor((1.0, 0.0, 0.0, 0.0)).unsqueeze(0)
                zero = action.new_zeros(1)
                turn = quat_from_euler_xyz(zero, zero, action.new_tensor([math.radians(yaw_deg)]))
                return quat_mul(turn, down)[0]

            grasp_quat = yaw_grasp_quat(args.hand_yaw_deg)

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
                if corner_indices is not None:
                    # Row written after the step, with the command actually sent (teleop convention)
                    episode_log.append(step_index * base.physics_dt, phase, action[0],
                                       as_tensor(robot.data.joint_pos)[0], corners(), anchor(),
                                       picker.is_grasping)
                # Correction approach: stop if the cloth already folded itself (all pairs in tolerance)
                # Only while the hand is still on its way (approach/descend), never once the fingers close
                if (phase.startswith("correction_") and phase.endswith(("_approach", "_descend"))
                        and pairs is not None and picker is not None
                        and not picker.is_grasping and step_index % 8 == 0):
                    folded = all(d <= args.success_distance for d in pair_distances(corners(), pairs))
                    transport["ok_streak"] = transport.get("ok_streak", 0) + 1 if folded else 0
                    if transport["ok_streak"] >= 4:
                        raise CorrectionNotNeeded(phase)
                # Physical grasp: watch the carried corner (corrections use a deliberately low arc)
                if (args.grasp_mode == "physical" and picker.is_grasping and "fold_arc" in phase
                        and not phase.startswith("correction")):
                    drift = picker.drift()
                    transport["max_drift_m"] = max(transport["max_drift_m"], drift)
                    if drift > args.slip_drift and transport["slip_phase"] is None:
                        transport["slip_phase"] = phase
                    # Dropped: the corner lies on the table while the hand is still well above it
                    # (at the end of the arc both come down together, which is not a drop)
                    held_z = (float(as_tensor(cloth.data.nodal_pos_w)[0, transport["pinch_vertex"], 2])
                              if "pinch_vertex" in transport else float(picker.target_position_w[2]))
                    hand_limit = transport.get("hand_ref_z", table_top_z + args.drop_hand_height)
                    if held_z - table_top_z < args.drop_height and float(anchor()[2]) > hand_limit:
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
                if picker is not None and not picker.is_grasping and args.free_speed > 0:
                    count = max(1, round(count / args.free_speed))  # nothing held: move faster
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
                """Where the TCP goes down: the corner itself (virtual) or the cloth point just inside it (physical)."""
                if physical:
                    # A fixed piece of cloth (chosen on the flat cloth), read where it is now: still just inside
                    # the corner when the cloth is folded, flipped or rotated, and it is on the same layer.
                    return as_tensor(cloth.data.nodal_pos_w)[0, pinch_vertices[source]].clone()
                return corners()[source].clone()

            def grasp_check(source, number):
                """Lift a little and hold: the corner must follow and stay close to the TCP."""
                nonlocal phase
                phase = f"{number}_grasp_check"
                # Let the squeeze settle before lifting (the close command itself already took 1 s)
                hold(math.ceil(args.grasp_settle_s / base.physics_dt))
                # Physical: judge the pinched vertex (about 4 cm inside the corner); the corner tip sags
                probe = ((lambda: as_tensor(cloth.data.nodal_pos_w)[0, pinch_vertices[source]]) if physical
                         else (lambda: corners()[source]))
                start_z = float(probe()[2])
                corner_start_z = float(corners()[source][2])
                lift = anchor().clone()
                lift[2] += args.check_lift
                # Slow lift: smoothstep over check_lift_s, peak speed = 1.5 * check_lift / check_lift_s
                move(lift, max(1, math.ceil(args.check_lift_s / base.physics_dt)), strict=False)
                hold(math.ceil(args.check_hold_s / base.physics_dt))
                corner = corners()[source]
                rise = float(probe()[2]) - start_z
                if physical:
                    transport["pinch_vertex"] = int(pinch_vertices[source])
                    transport["hand_ref_z"] = float(anchor()[2])
                corner_rise = float(corner[2]) - corner_start_z
                corner_tcp = float(torch.linalg.vector_norm(corner - anchor()))
                ok = rise >= args.check_min_rise and (args.quality_max <= 0 or corner_tcp <= args.quality_max)
                return {"rise_m": rise, "corner_rise_m": corner_rise, "corner_tcp_m": corner_tcp, "ok": ok}

            def place_corner(source, destination, number, correction=False):
                nonlocal phase
                vertex = int(corner_indices[source])
                record = {"number": str(number), "source": int(source), "attempts": 0, "ok": False}
                approach_quat = grasp_quat
                if physical and correction and edge_vertices is not None:
                    # The picked corner may lie on a folded layer turned away from the start rotation:
                    # read that layer's edge direction now and turn the hand to it
                    nodes_now = as_tensor(cloth.data.nodal_pos_w)[0]
                    ex, ey = edge_vertices[source]
                    layer = layer_jaw_yaw(nodes_now[int(corner_indices[source]), :2], nodes_now[ex, :2], nodes_now[ey, :2])
                    hand = math.remainder(args.hand_yaw_deg + math.degrees(layer), 180.0)
                    approach_quat = yaw_grasp_quat(hand)
                    record.update(layer_yaw_deg=round(math.degrees(layer), 2), hand_yaw_deg=round(hand, 2))
                    print(f"[fold] correction {number}: layer yaw={math.degrees(layer):.2f} deg, hand yaw={hand:.2f} deg",
                          flush=True)
                result["grasps"].append(record)
                for attempt in range(1 + (args.retries if physical else 0)):
                    record["attempts"] = attempt + 1
                    phase = f"{number}_approach"
                    above = grasp_target(source)
                    above[2] += 0.08
                    # Raise vertically before moving across the cloth.
                    safe = anchor().clone()
                    safe[2] = max(float(safe[2]), float(above[2]))
                    if attempt == 0 or not args.retry_in_place:
                        move(safe, 120)
                        if args.approach_order == "yx":
                            # Sideways (world y, as the robot sees it) first at the safe height, then x
                            via = safe.clone()
                            via[1] = above[1]
                            move(via, 120, goal_quat_w=approach_quat)
                            move(above, 120)
                        else:
                            move(above, goal_quat_w=approach_quat)
                    # Gap in free air, to tell "never opened fully" from "squeezed while descending"
                    record["open_gap_above_m"] = float(as_tensor(robot.data.joint_pos)[0, finger_ids].sum())
                    if len(finger_body_ids) == 2:
                        # World vector between the two fingers: its larger component is the real jaw axis
                        fingers = as_tensor(robot.data.body_pos_w)[0, finger_body_ids]
                        record["finger_vec_open_mm"] = [round(float(v) * 1000, 1) for v in fingers[0] - fingers[1]]
                        print(f"[fold] finger vector (open, world) mm={record['finger_vec_open_mm']}, "
                              f"expected jaw axis={args.jaw_axis}", flush=True)
                    if "finger_pad_probe" not in result:
                        # Pad size from the collision geometry (once per run); replaces the pad_half estimate
                        result["finger_pad_probe"] = probe_finger_pads()
                        print(f"[fold] finger pad probe: {json.dumps(result['finger_pad_probe'])}", flush=True)
                    phase = f"{number}_descend"
                    touch = grasp_target(source)
                    # Virtual: stop 1.5 cm above the corner. Physical: fingertips on the cloth (blocked by the table)
                    touch[2] += -args.press_depth if physical else 0.015
                    move(touch, 180, strict=not physical)
                    if physical:
                        # Physical descent is not strict: record where the hand really stopped vs the target
                        record["descend_error_mm"] = [round(float(v) * 1000, 1) for v in anchor() - touch]
                    if physical and args.close_lift > 0:
                        # Fingertips pressed on the table cannot close; lift off first
                        lifted = anchor().clone()
                        lifted[2] += args.close_lift
                        move(lifted, 60, strict=False)
                        record["close_lift_mm"] = round(args.close_lift * 1000, 1)
                    phase = f"{number}_grasp"
                    record["open_gap_m"] = float(as_tensor(robot.data.joint_pos)[0, finger_ids].sum())
                    # Dataset: pick = TCP xy at close (physical) or the corner (virtual), as in teleop
                    episode_log.begin(corners(), anchor()[:2] if physical else corners()[source, :2])
                    set_gripper(closed=True)
                    record["gap_closed_mm"] = round(float(as_tensor(robot.data.joint_pos)[0, finger_ids].sum()) * 1000, 1)
                    current = corners()[source].clone()
                    if not picker.grasp(current, vertex_id=vertex):
                        if not physical:
                            raise RuntimeError(f"Corner {source} is outside grasp radius")
                        record.update(ok=False, reason="gripper_not_near_corner")
                    elif not physical:
                        record.update(ok=True)
                        episode_log.start_carry()
                    else:
                        episode_log.start_carry()  # the grasp-check lift is the start of the learned carry
                        record["offset_at_close_mm"] = [round(float(v) * 1000, 1) for v in picker.corner_offset()]
                        record.update(grasp_check(source, number))
                        if record["ok"]:
                            # Measure slip from the settled grip, not from the moment of closing
                            picker.grasp_offset = picker.corner_offset().clone()
                    print(f"[fold] grasp {number} attempt {attempt + 1}: {record}", flush=True)
                    if record["ok"]:
                        break
                    episode_log.abandon()  # failed attempt: kept as grasp_retry, not trained as carry
                    picker.release()
                    set_gripper(closed=False)
                    if not args.retry_in_place:
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
                # mirror: the hand ends where it would pinch the destination corner (the grasp point is
                # mirrored by the fold), so the held corner lands on the destination instead of overshooting it.
                mirror = physical and args.place_mode == "mirror"
                tcp_start_xy = anchor()[:2].clone()
                # Unit vector source -> destination; the place point is pulled back along it (toward the fold line)
                place_dir = corners()[destination, :2] - start_node[:2]
                place_dir = place_dir / torch.linalg.vector_norm(place_dir).clamp_min(1e-6)
                place_shift = -args.place_inset * place_dir
                # The grasp check already lifted the corner by rise_m. Build the arc from the height the
                # corner had before that lift, so the check is only a prefix of the carry and the arc is
                # not stacked on top of it (same path as a run without the check).
                arc_start = start_node.clone()
                if physical and not args.stack_check_lift:
                    arc_start[2] -= record.get("corner_rise_m", record.get("rise_m", 0.0))
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
                    node_goal = torch.lerp(arc_start, destination_now, alpha)
                    if args.arc_peak is not None and not correction:
                        # Parabola: starts where the grasp check left the corner, peaks at arc_peak
                        # above the pre-lift height, ends just above the destination corner.
                        z0 = float(start_node[2])
                        z1 = (table_top_z + args.release_height + float(offset[2]) if args.release_height is not None
                              else float(destination_now[2]) + max(0.003, args.cloth_thickness + 0.001))
                        bump = max(0.0, float(arc_start[2]) + args.arc_peak - (z0 + z1) / 2)
                        node_goal[2] = z0 + (z1 - z0) * alpha + 4 * bump * t * (1 - t)
                    else:
                        node_goal[2] += arc_height * math.sin(math.pi * t) + max(0.003, args.cloth_thickness + 0.001) * alpha
                    goal = node_goal - offset
                    if mirror:
                        goal[:2] = torch.lerp(tcp_start_xy, grasp_target(destination)[:2] + place_shift, alpha)
                    if physical:
                        # goal = held point - offset can fall below the table when a folded layer is re-grasped
                        # (held point above the TCP); the fingers then stop on the table/cloth. Aim no lower than
                        # min_tcp_height and, like the lowering phase, go as far as possible instead of aborting.
                        goal[2] = max(float(goal[2]), table_top_z + args.min_tcp_height)
                    move(goal, 60 if correction else 120, strict=not physical)
                phase = f"{number}_lower"
                for _ in range(3):
                    if physical:
                        offset = picker.target_position_w - anchor()
                    goal = corners()[destination].clone() - offset
                    goal[2] += max(0.003, args.cloth_thickness + 0.001)
                    if args.release_height is not None:
                        goal[2] = table_top_z + args.release_height  # open at this TCP height
                    if mirror:
                        goal[:2] = grasp_target(destination)[:2] + place_shift
                    # Physical: fingers collide with the cloth layers under them, so the TCP cannot
                    # reach this height. Go as low as possible; success is judged by the corner
                    # distance after release, not by the TCP height.
                    move(goal, 120, strict=not physical)
                    if physical:
                        record["lower_error_m"] = float(torch.linalg.vector_norm(anchor() - goal))
                phase = f"{number}_release"
                record["max_drift_m"] = transport["max_drift_m"]

                def placed_error_mm():
                    """Signed xy of (held corner - destination corner) in mm: where the miss comes from."""
                    return [round(float(v) * 1000, 1) for v in (corners()[source] - corners()[destination])[:2]]

                record["err_before_open_mm"] = placed_error_mm()
                episode_log.begin_release(anchor()[:2])  # place = TCP xy right before opening
                set_gripper(closed=False)
                record["err_after_open_mm"] = placed_error_mm()
                picker.release()
                pair_index = next(i for i, p in enumerate(pairs) if tuple(p) == (source, destination))
                episode_log.end(corners(), pair_index, correction=correction)
                phase = f"{number}_retreat"
                retreat = anchor().clone()
                retreat[2] += 0.10
                move(retreat, 180)
                wait_s = args.settle_seconds if args.place_settle_s is None else args.place_settle_s
                hold(max(1, math.ceil(wait_s / base.physics_dt)))
                record["err_settled_mm"] = placed_error_mm()
                print(f"[fold] placement {number} corner error mm (x,y): before open={record['err_before_open_mm']}, "
                      f"after open={record['err_after_open_mm']}, settled={record['err_settled_mm']}", flush=True)
                print(f"[fold] after placement {number}: distances={pair_distances(corners(), pairs)}", flush=True)


            with torch.inference_mode():
                hold(120)
                if args.start_yaw_deg != 0.0:
                    # Rotate the settled flat cloth as a whole, at rest, then let it settle again
                    rotated = rotate_nodes_about_center(as_tensor(cloth.data.nodal_pos_w)[0],
                                                        math.radians(args.start_yaw_deg)).unsqueeze(0)
                    for name, values in (("write_nodal_pos_to_sim", rotated.contiguous()),
                                         ("write_nodal_velocity_to_sim", torch.zeros_like(rotated))):
                        writer = getattr(cloth, name + "_index", None) or getattr(cloth, name)
                        writer(values, env_ids=[0])
                    hold(120)
                corner_indices = detect_corner_vertex_indices(as_tensor(cloth.data.nodal_pos_w)[0])
                # Measured cloth rotation (edges vs world x), used as the cloth axes from here on
                cloth_yaw = cloth_yaw_from_corners(corners()) if physical else 0.0
                result["cloth_yaw_measured_deg"] = math.degrees(cloth_yaw)
                pairs = rectangular_pairs(corners(), as_tensor(robot.data.root_pos_w)[0, :2])
                if args.move_axis != "auto":
                    # x / y are the cloth's own axes (the world axes when the cloth is not rotated)
                    pairs = axis_pairs(corners(), as_tensor(robot.data.root_pos_w)[0, :2], args.move_axis, cloth_yaw)
                if physical:
                    # Jaw closes along the cloth axis: hand yaw = grasp_yaw_deg + cloth rotation,
                    # kept within +-90 deg because the gripper is 180 deg symmetric
                    hand_yaw = math.remainder(args.hand_yaw_deg + math.degrees(cloth_yaw), 180.0)
                    result["hand_yaw_used_deg"] = hand_yaw
                    grasp_quat = yaw_grasp_quat(hand_yaw)
                    print(f"[fold] cloth yaw measured={math.degrees(cloth_yaw):.2f} deg "
                          f"(commanded {args.start_yaw_deg:.2f}), hand yaw={hand_yaw:.2f} deg", flush=True)
                initial_corners = corners().clone()
                initial_center_xy = corners()[:, :2].mean(dim=0).clone()
                result["initial_center_xy"] = initial_center_xy.tolist()
                if physical:
                    flat_nodes = as_tensor(cloth.data.nodal_pos_w)[0].clone()
                    pinch_vertices = [pinch_vertex(flat_nodes[:, :2], int(v), initial_center_xy, args.open_width,
                                                   pad_half=args.pad_half, edge_margin=args.edge_margin,
                                                   jaw_axis=args.jaw_axis, jaw_shift=args.pinch_jaw_shift,
                                                   axes_yaw=cloth_yaw) for v in corner_indices.tolist()]
                    result["pinch_vertices"] = pinch_vertices
                    # Per corner: cloth points 6 cm along its two edges, to read a layer's direction later
                    edge_vertices = [edge_reference_vertices(flat_nodes[:, :2], int(v), initial_center_xy, cloth_yaw)
                                     for v in corner_indices.tolist()]
                    result["edge_reference_vertices"] = edge_vertices
                    # Achieved offset from each corner (mesh spacing limits how exact it is)
                    result["pinch_offsets_mm"] = [
                        [round(float(d) * 1000, 1) for d in flat_nodes[p, :2] - flat_nodes[c, :2]]
                        for p, c in zip(pinch_vertices, corner_indices.tolist())]
                    # The same offsets along the cloth's own axes (equal to the above when it is not rotated)
                    result["pinch_offsets_cloth_mm"] = [
                        [round(float(d) * 1000, 1) for d in to_cloth_frame(flat_nodes[p, :2], flat_nodes[c, :2], cloth_yaw)]
                        for p, c in zip(pinch_vertices, corner_indices.tolist())]
                    print(f"[fold] pinch vertices={pinch_vertices}, offsets from corner mm={result['pinch_offsets_mm']}",
                          flush=True)
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
                    # Keep the original intermediate-stage evaluation unchanged.
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
                    transport["ok_streak"] = 0
                    try:
                        place_corner(*pairs[worst], number=f"correction_{correction + 1}", correction=True)
                    except CorrectionNotNeeded as stop:
                        phase = "correction_stopped"
                        result["correction_skipped_phase"] = stop.phase
                        if picker.is_grasping:
                            picker.release()
                        if float(action[0, 7]) < 0:
                            set_gripper(closed=False)  # never lift the cloth while stopping
                        print(f"[fold] cloth already folded during {stop.phase}; stopping the correction", flush=True)
                        up = anchor().clone()
                        up[2] += 0.08
                        move(up, 120, strict=False)
                        break
                    result["correction_count"] += 1
                phase = "evaluation"
                hold(math.ceil(args.settle_seconds / base.physics_dt))

                # Keep the original CLI-configurable pair-distance criterion.
                distances = pair_distances(corners(), pairs)
                corner_ok = all(
                    math.isfinite(d) and d <= args.success_distance for d in distances
                )

                # Evaluate the final folded silhouette using the separate IoU metric.
                shape_ok, shape_metrics = fold_shape_success(
                    nodes_w=as_tensor(cloth.data.nodal_pos_w)[0],
                    initial_corners=initial_corners,
                    current_corners=corners(),
                    pairs=pairs,
                    min_iou=args.shape_iou_threshold,
                    cloth_size=args.cloth_size,
                )
                result.update(
                    status="completed",
                    final_distances_m=distances,
                    final_corner_positions_w=corners().tolist(),
                    final_rms_speed_m_s=float(torch.sqrt(
                        (as_tensor(cloth.data.nodal_vel_w)[0] ** 2).sum(-1).mean()
                    )),
                    corner_success=corner_ok,
                    shape_success=shape_ok,
                    shape_metrics=shape_metrics,
                    success=corner_ok and shape_ok,
                )
                print(
                    f"[fold] success={result['success']}, corner_ok={corner_ok}, "
                    f"IoU={shape_metrics['iou']:.4f}, shape_ok={shape_ok}, "
                    f"final distances={distances}",
                    flush=True,
                )
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
                # Put the outcome in the folder name: ok / fail / dropped / grasp_failed / error
                tag = ("ok" if result["success"] else
                       "fail" if result["status"] == "completed" else result["status"])
                tagged = run_dir.with_name(f"{run_dir.name}_{tag}")
                try:
                    run_dir.rename(tagged)
                    run_dir = tagged
                    if video_dir is not None:
                        video_dir = tagged
                except OSError as exc:
                    print(f"[fold] could not rename the run folder: {exc}", flush=True)
                result["run_tag"] = tag
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
                if (result["success"] and result["status"] == "completed" and args.stop_after == "fold"
                        and "final_corner_positions_w" in result):
                    # BC dataset in the PR #37 format (high_level.jsonl, low_level.npz, metadata.json, context.json)
                    try:
                        context = fold_context(
                            grasp_mode=args.grasp_mode, corner_vertex_indices=result["corner_vertex_ids"],
                            pairs=pairs, sample_dt=base.physics_dt, success_threshold_m=args.success_distance,
                            final_pair_distances_m=result["final_distances_m"], config=vars(args))
                        saved = episode_log.save(run_dir, pairs, args.success_distance,
                                                 result["final_corner_positions_w"], run_dir.name, context)
                        print(f"[fold] BC dataset {'saved' if saved else 'NOT saved'}: {run_dir} "
                              f"({len(episode_log.segments)} pick/place, {len(episode_log.rows)} steps)", flush=True)
                    except Exception as exc:  # never lose the run because of the export
                        print(f"[fold] BC dataset export failed: {type(exc).__name__}: {exc}", flush=True)
            if video_dir is not None and result["success"]:
                clips = list(video_dir.glob("*.mp4"))
                if not clips or not all(clip.stat().st_size > 0 for clip in clips):
                    raise RuntimeError(f"No video was saved; check the recording errors above: {video_dir}")
                for clip in clips:
                    print(f"[A_single_point] Video saved: {clip}", flush=True)
            elif video_dir is not None:
                print("[fold] Unsuccessful trial: video frames discarded without saving MP4.", flush=True)
            if video_dir is not None:
                # Same tag at the front of the video name, so a copied MP4 still says how the run ended
                for clip in video_dir.glob("rectangular_fold_*.mp4"):
                    clip.rename(clip.with_name(f"{result.get('run_tag', 'unknown')}_{clip.name}"))


if __name__ == "__main__":
    main()
