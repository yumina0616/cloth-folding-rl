
"""Sequential rectangular cloth fold with one virtual gripper.

Run with the Isaac Lab launcher; see README_rectangular_fold.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import cv2
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from datetime import datetime
import math
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
import isaaclab_tasks  

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.learning.demonstration_recorder import DemonstrationRecorder

from a_single_point_picker import as_tensor
from virtual_attachment_picker import create_picker, filter_gripper_contacts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions
from rectangular_fold_geometry import rectangular_pairs, pair_distances
from successful_video import gate_video_recorders
from towel_visual import TowelVisual
def filter_gripper_contacts_event(env, env_ids):
    """Isaac Lab event callback for virtual grasp collision filtering."""
    return filter_gripper_contacts(env)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture_limit", type=int, default=15,
                        help="Maximum synchronized camera pairs per run")
    parser.add_argument("--capture_stride", type=int, default=75,
                        help="Minimum physics steps between captures")
    parser.add_argument("--capture_size", type=int, default=640)
    parser.add_argument("--capture_dir", type=Path, default=project_root / "data" / "vision" / "raw")
    parser.add_argument("--wrist_parent", type=str, default="{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_hand",
                        help="Exact hand USD prim path; expected nested Geometry structure in Lab 3.0")
    parser.add_argument("--grasp_mode", choices=["virtual"], default="virtual")
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
    parser.add_argument("--video_dir", type=Path, default=Path(__file__).resolve().parents[2] / "videos" / "rectangular_fold")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    args.video = not args.no_video
    args.enable_cameras = True
    if args.capture_limit < 1 or args.capture_stride < 1:
        parser.error("capture_limit and capture_stride must be >=1")
    for name in ("lift_height", "settle_seconds", "success_distance", "bend_stiffness",
                 "cloth_size", "mesh_refinement", "table_length", "table_width", "youngs_modulus",
                 "cloth_thickness", "cloth_density"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.mesh_refinement < 1:
        parser.error("mesh_refinement must be at least 1")
    if not all(math.isfinite(v) for v in (args.cloth_x, args.cloth_y, args.cloth_yaw_deg)):
        parser.error("Cloth pose must be finite")
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    material = cfg.scene.deformable.spawn.physics_material
    original_bend = material.surface_bend_stiffness
    material.surface_bend_stiffness = args.bend_stiffness
    material.youngs_modulus = args.youngs_modulus
    material.surface_thickness = args.cloth_thickness
    material.density = args.cloth_density
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    print(f"[fold] cloth={args.cloth_size}m, mesh refinement={args.mesh_refinement}, "
          f"bend stiffness={original_bend:g}->{args.bend_stiffness:g}, Young modulus={args.youngs_modulus:g}", flush=True)
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # V3.0 CameraCfg.OffsetCfg uses xyzw quaternion ordering.
    # OpenGL camera with identity local orientation looks along -Z.
    cfg.scene.top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/GlobalTopCamera",
        update_period=0.0, height=args.capture_size, width=args.capture_size,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.03, 10.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(args.cloth_x, args.cloth_y, 1.55),
            rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
    )
    # True eye-in-hand: camera is a CHILD prim of panda_hand, not TCP-following.
    # The prim inherits the translation AND rotation of its parent hand link.
    cfg.scene.wrist_camera = CameraCfg(
        prim_path=args.wrist_parent.rstrip("/") + "/WristCamera",
        update_period=0.0, height=args.capture_size, width=args.capture_size,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.02, 10.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.04, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    )
    # update() is called once per env.step, so step at the physics frequency.
    cfg.decimation = 1
    cfg.episode_length_s = 300.0
    cfg.scene.support_neg_y = None
    cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.size = (args.table_length, args.table_width, 1.05)
    cfg.scene.table.spawn.visible = True
    cfg.scene.deformable.init_state.pos = (args.cloth_x, args.cloth_y, 0.012)
    half_yaw = math.radians(args.cloth_yaw_deg) / 2
    cfg.scene.deformable.init_state.rot = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cfg.events.reset_deformable = None
    cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts_event, mode="prestartup")
    print("[DEBUG] virtual_grasp_contacts params:",
          cfg.events.virtual_grasp_contacts.params, flush=True)
    # Repeat the same starting arm pose for comparable vibration observations.
    cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
    cfg.curriculum.gravity = None
    cfg.curriculum.action_rate = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    cfg.actions = type(cfg)().actions.ik
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

    demo_dir = project_root / "data" / "demonstrations" / run_dir.name

    
    recorder = None
    capture_root = args.capture_dir.resolve() / run_dir.name
    capture_root.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    last_periodic_capture_step = -10**9
    category_counts = {"flat": 0, "lifted": 0, "partial": 0, "folded": 0}
    # Reserve up to six shots for approach/descend/grasp on the first two corners.
    # The remaining slots are for periodic captures across the folding phases.
    event_limit = min(6, args.capture_limit)
    periodic_limit = args.capture_limit - event_limit
    event_saved = 0
    periodic_saved = 0
    fractions = (0.27, 0.27, 0.26)
    quotas = {"flat": round(periodic_limit * fractions[0]),
              "lifted": round(periodic_limit * fractions[1]),
              "partial": round(periodic_limit * fractions[2])}
    quotas["folded"] = periodic_limit - sum(quotas.values())


    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    result = {"success": False, "status": "running", "threshold_m": args.success_distance}
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
        gate_video_recorders(env, lambda: result["status"] == "completed" and result["success"])
        picker = None
        try:
            env.reset()
            base = env.unwrapped
            # Success markers are created visible independently of debug_vis.
            # With no success asset they otherwise spawn a table-sized cube at
            # the world origin, hiding the robot and cloth.
            term = base.command_manager.get_term("deformable_pose")
            visualizer = term.success_visualizer

            if visualizer is not None:
                visualizer.set_visibility(False)
            scene = base.scene
            cloth, ee = scene["deformable"], scene["ee_frame"]
            towel = TowelVisual(cloth, args.cloth_thickness,
                                Path(__file__).resolve().parents[2] / "assets" / "materials" / "towel")
            result["mesh_node_count"] = int(as_tensor(cloth.data.nodal_pos_w).shape[1])
            print(f"[fold] mesh node count={result['mesh_node_count']}", flush=True)
            robot = scene["robot"]
            cameras = {"global": scene["top_camera"], "wrist": scene["wrist_camera"]}
            from pxr import Usd
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            expected = args.wrist_parent.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            if not stage.GetPrimAtPath(expected).IsValid():
                options = [p.GetPath().pathString for p in stage.Traverse()
                           if p.GetName() in ("panda_hand", "panda_link7")]
                raise RuntimeError(
                    f"Wrist parent not found: {expected}. Candidate paths={options}. "
                    "Pass --wrist_parent <the correct panda_hand path>.")
            print(f"[capture] wrist parent: {expected}", flush=True)
            result["filtered_gripper_contacts"] = base.virtual_gripper_filtered_paths
            finger_ids = robot.find_joints("panda_finger_joint.*")[0]
            if len(finger_ids) != 2:
                raise RuntimeError("Expected two Franka finger joints for gripper verification")
            anchor = lambda: as_tensor(ee.data.target_pos_w)[0, 0]
            picker = create_picker(cloth, anchor, grasp_mode=args.grasp_mode, grasp_radius=args.grasp_radius)
            action = torch.zeros(base.action_space.shape, device=base.device)
            if action.shape != (1, 8):
                raise RuntimeError(f"Expected absolute pose IK action (1, 8), got {action.shape}")
            target_w = anchor().clone()
            target_quat_w = as_tensor(ee.data.target_quat_w)[0, 0].clone()

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

            def _numpy(value):
                if hasattr(value, "torch"):
                    value = value.torch
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value)

            def _quat_xyzw_to_rot(q):
                # Isaac Lab 3.0 ROS camera pose uses quaternion XYZW.
                x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
                return np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])

            def _phase_category(current_phase):
                name = str(current_phase)
                if name in ("evaluation", "correction_evaluation"):
                    return "folded"
                if name.startswith("2_") or name.startswith("correction_"):
                    return "partial"
                if "fold_arc" in name or "lower" in name:
                    return "lifted"
                return "flat"

            def save_camera_sample(capture_event=None, target_corner_index=None,
                                   destination_corner_index=None):
                nonlocal saved_count, last_periodic_capture_step, event_saved, periodic_saved
                if corner_indices is None or saved_count >= args.capture_limit:
                    return False
                category = _phase_category(phase)
                is_event = capture_event is not None
                if is_event:
                    # Event shots bypass time spacing and per-phase quotas.
                    if event_saved >= event_limit:
                        return False
                    if target_corner_index is None:
                        raise ValueError("Event capture requires target_corner_index")
                else:
                    if periodic_saved >= periodic_limit:
                        return False
                    if category_counts[category] >= quotas[category]:
                        return False
                    if step_index - last_periodic_capture_step < args.capture_stride:
                        return False
                xyz = _numpy(corners()).reshape(4, 3)
                basename = f"frame_{saved_count:04d}"
                record = {
                    "frame_id": basename, "physics_step": step_index,
                    "phase": phase, "category": category,
                    "capture_kind": "event" if is_event else "periodic",
                    "capture_event": capture_event,
                    "target_corner_index": int(target_corner_index) if target_corner_index is not None else None,
                    "destination_corner_index": int(destination_corner_index) if destination_corner_index is not None else None,
                    "cloth_start_xy_yaw": [args.cloth_x, args.cloth_y, args.cloth_yaw_deg],
                    "corner_vertex_ids": corner_indices.tolist(),
                    "corners_world_xyz_m": xyz.tolist(),
                    "labels_verified": False,
                    "occlusion_verified": False,
                    "cameras": {},
                }
                pending = []
                for name, cam in cameras.items():
                    rgb = _numpy(cam.data.output["rgb"])[0]
                    rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
                    K = _numpy(cam.data.intrinsic_matrices)[0]
                    camera_pos = _numpy(cam.data.pos_w)[0]
                    camera_quat_ros = _numpy(cam.data.quat_w_ros)[0]
                    R = _quat_xyzw_to_rot(camera_quat_ros)
                    # World points -> ROS optical camera coordinates.
                    cam_xyz = (xyz - camera_pos) @ R
                    uv = []
                    in_frame = []
                    h, w = rgb.shape[:2]
                    for x, y, z in cam_xyz:
                        valid = bool(z > 0.001)
                        u = float(K[0,0]*x/z + K[0,2]) if valid else -1.0
                        v = float(K[1,1]*y/z + K[1,2]) if valid else -1.0
                        uv.append([u, v])
                        in_frame.append(valid and 0 <= u < w and 0 <= v < h)
                    overlay = rgb.copy()
                    for i, (u, v) in enumerate(uv):
                        if in_frame[i]:
                            xpix, ypix = int(round(u)), int(round(v))
                            cv2.circle(overlay, (xpix, ypix), 5, (255, 0, 0), -1)
                            cv2.putText(overlay, str(i), (xpix+7, ypix-7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    image_name = f"{basename}_{name}.png"
                    overlay_name = f"{basename}_{name}_overlay.png"
                    pending += [(image_name, rgb), (overlay_name, overlay)]
                    record["cameras"][name] = {
                        "image": image_name, "overlay": overlay_name,
                        "intrinsic_matrix": K.tolist(),
                        "camera_pos_world": camera_pos.tolist(),
                        "camera_quat_ros_xyzw": camera_quat_ros.tolist(),
                        "keypoints_pixel_unverified": uv,
                        "in_frame_not_visibility": in_frame,
                    }
                for filename, rgb in pending:
                    ok = cv2.imwrite(str(capture_root / filename),
                                     cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise RuntimeError(f"Failed writing image: {filename}")
                (capture_root / f"{basename}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                saved_count += 1
                if is_event:
                    event_saved += 1
                else:
                    periodic_saved += 1
                    category_counts[category] += 1
                    last_periodic_capture_step = step_index
                print(f"[capture] {saved_count}/{args.capture_limit} "
                      f"kind={'event' if is_event else 'periodic'} event={capture_event} "
                      f"target={target_corner_index} phase={phase}", flush=True)
                return True

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
                if corner_indices is not None and step_index % 15 == 0:
                    save_camera_sample()
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

            def move(goal_w, count=240, goal_quat_w=None, midpoint_callback=None):
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

                    if midpoint_callback is not None and i == count // 2:
                        midpoint_callback()
                target_w = goal_w.clone()
                target_quat_w = goal_quat
                # Fixed-duration interpolation alone does not guarantee arrival.
                for _ in range(1200):
                    step()
                    if float(torch.linalg.vector_norm(anchor() - goal_w)) < 0.005:
                        break
                error = float(torch.linalg.vector_norm(anchor() - goal_w))
                print(f"[virtual_grasp] move goal={goal_w.tolist()}, actual={anchor().tolist()}, error={error:.4f} m", flush=True)
                if error >= 0.005:
                    raise RuntimeError(f"IK did not reach target: {error:.4f} m; goal={goal_w.tolist()}, actual={anchor().tolist()}")

            def place_corner(source, destination, number, correction=False):
                nonlocal phase
                vertex = int(corner_indices[source])
                phase = f"{number}_approach"
                above = corners()[source].clone()
                above[2] += 0.08
                # Raise vertically before moving across the cloth.
                safe = anchor().clone()
                safe[2] = max(float(safe[2]), float(above[2]))
                move(safe, 120)
                move(above, goal_quat_w=action.new_tensor((1.0, 0.0, 0.0, 0.0)))
                if not correction:
                    save_camera_sample("approach", source, destination)
                phase = f"{number}_descend"
                touch = corners()[source].clone()
                touch[2] += 0.015
                if not correction:
                    move(
                        touch,
                        180,
                        midpoint_callback=lambda: save_camera_sample(
                            "descend", source, destination
                        ),
                    )
                else:
                    move(touch, 180)
                phase = f"{number}_grasp"
                set_gripper(closed=True)
                current = corners()[source].clone()
                if not picker.grasp(current, vertex_id=vertex):
                    raise RuntimeError(f"Corner {source} is outside grasp radius")
                if not correction:
                    save_camera_sample("grasp", source, destination)
                # Sweep around the fold line. A high horizontal translation
                # pulled the far edge away and exceeded the robot workspace.
                offset = picker.target_position_w - anchor()
                start_node = corners()[source].clone()
                initial_span = float(torch.linalg.vector_norm(corners()[destination, :2] - start_node[:2]))
                arc_height = (max(0.008, min(args.lift_height, initial_span / 2)) if correction
                                  else max(args.lift_height, initial_span / 2))
                for segment in range(1, 9):
                    phase = f"{number}_fold_arc_{segment}"
                    t = segment / 8
                    alpha = (1 - math.cos(math.pi * t)) / 2
                    destination_now = corners()[destination].clone()
                    node_goal = torch.lerp(start_node, destination_now, alpha)
                    node_goal[2] += arc_height * math.sin(math.pi * t) + max(0.003, args.cloth_thickness + 0.001) * alpha
                    move(node_goal - offset, 60 if correction else 120)
                phase = f"{number}_lower"
                for _ in range(3):
                    goal = corners()[destination].clone() - offset
                    goal[2] += max(0.003, args.cloth_thickness + 0.001)
                    move(goal, 120)
                phase = f"{number}_release"
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
                recorder = DemonstrationRecorder(
                demo_dir=demo_dir,
                pairs=pairs,
                success_threshold=args.success_distance,
            )
                result["corner_vertex_ids"] = corner_indices.tolist()
                result["pairs"] = pairs
                result["initial_distances_m"] = pair_distances(corners(), pairs)
                print(f"[fold] tracked vertices={corner_indices.tolist()}, pairs={pairs}", flush=True)

                # Record each high-level Pick & Place without changing the fold controller.
                for number, (source, destination) in enumerate(pairs, 1):
                    low_level_start = len(rows)
                    before_corners = corners().clone()
                    pick_xy = before_corners[source, :2].cpu().tolist()
                    place_xy = before_corners[destination, :2].cpu().tolist()

                    place_corner(source, destination, number)
                    low_level_end = len(rows)
                    after_corners = corners().clone()
                    reward, success = recorder.record_action(
                        before_corners=before_corners,
                        after_corners=after_corners,
                        pick_xy=pick_xy,
                        place_xy=place_xy,
                        low_level_start=low_level_start,
                        low_level_end=low_level_end,
                    )

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
                    source, destination = pairs[worst]
                    low_level_start = len(rows)
                    before_corners = corners().clone()
                    pick_xy = before_corners[source, :2].cpu().tolist()
                    place_xy = before_corners[destination, :2].cpu().tolist()

                    place_corner(
                        source, destination,
                        number=f"correction_{correction + 1}", correction=True,
                    )
                    low_level_end = len(rows)
                    after_corners = corners().clone()
                    reward, success = recorder.record_action(
                        before_corners=before_corners,
                        after_corners=after_corners,
                        pick_xy=pick_xy,
                        place_xy=place_xy,
                        low_level_start=low_level_start,
                        low_level_end=low_level_end,
                    )
                    result["correction_count"] += 1

                phase = "evaluation"
                hold(math.ceil(args.settle_seconds / base.physics_dt))
                final_corners = corners().clone()
                distances = pair_distances(final_corners, pairs)
                result.update(
                    status="completed",
                    final_distances_m=distances,
                    final_corner_positions_w=final_corners.tolist(),
                    final_rms_speed_m_s=float(
                        torch.sqrt((as_tensor(cloth.data.nodal_vel_w)[0] ** 2).sum(-1).mean())
                    ),
                    success=all(math.isfinite(d) and d <= args.success_distance for d in distances),
                )
                # Final settle happens after the last Pick & Place: make the final
                # transition's next observation, reward and terminal status agree.
                if recorder is not None and recorder.logger.records:
                    recorder.finalize(
                        final_corners=final_corners,
                        success=result["success"],
                    )
                print(f"[fold] success={result['success']}, final distances={distances}", flush=True)


        except BaseException as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", phase=phase)
            raise
        finally:
            if picker is not None:
                picker.reset()
            try:
                env.close()
            finally:
                result["capture_count"] = saved_count
                result["capture_by_category"] = category_counts
                result["capture_event_count"] = event_saved
                result["capture_periodic_count"] = periodic_saved
                (capture_root / "capture_summary.json").write_text(
                    json.dumps({"saved_pairs": saved_count, "phase_counts": category_counts,
                                "saved_event_pairs": event_saved,
                                "saved_periodic_pairs": periodic_saved,
                                "reserved_event_pairs": event_limit,
                                "expected_pairs": args.capture_limit}, indent=2), encoding="utf-8")
                (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                if rows:
                    # Keep every trial's diagnostic measurements. Only successful
                    # trials are exported to the imitation-learning dataset.
                    with (run_dir / "steps.csv").open("w", newline="", encoding="utf-8") as output:
                        writer = csv.writer(output)
                        writer.writerow(["time_s", "phase", "grasping", "pair_1_distance_m", "pair_2_distance_m"])
                        for t, label, _, _, points, _, grasping in rows:
                            distances = pair_distances(torch.from_numpy(points), pairs)
                            writer.writerow([t, label, grasping, *distances])
                    if result["success"] and recorder is not None:
                        recorder.save_if_success(
                            success=result["success"],
                            low_level_data=rows,
                            episode_id=run_dir.name,
                        )
                        print(f"[logger] Saved demonstration to {demo_dir}", flush=True)

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
