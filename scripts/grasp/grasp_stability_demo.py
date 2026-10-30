"""Grasp-only stability diagnostic: compare solver attachment and nodal projection.

Run with the Isaac Lab launcher; see README_grasp_stability.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
import math
import sys
from pathlib import Path

# Isaac Lab imports its recorder while registering tasks, so expose optional
# project-local recording packages before importing Isaac Lab.
if "--video" in sys.argv:
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
import isaaclab_tasks  # noqa: F401

from a_single_point_picker import as_tensor
from successful_video import gate_video_recorders
from towel_visual import TowelVisual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picker_backend", choices=["attachment", "projection"], default="attachment")
    parser.add_argument("--grasp_mode", choices=["virtual"], default="virtual")
    parser.add_argument("--grasp_radius", type=float, default=0.03)
    parser.add_argument("--lift_height", type=float, default=0.08)
    parser.add_argument("--move_distance", type=float, default=0.0)
    parser.add_argument("--hold_steps", type=int, default=120)
    parser.add_argument("--observe_seconds", type=float, default=10.0,
                        help="Hold the lifted pose and record cloth motion for this many simulated seconds")
    parser.add_argument("--metrics_path", type=Path,
                        default=Path(__file__).with_name("grasp_stability_observation.csv"))
    parser.add_argument("--bend_stiffness", type=float, default=2e4 / 108,
                        help="Surface bend stiffness; experiment 02 uses 1e6")
    parser.add_argument("--video", action="store_true", help="Automatically save the full demonstration as MP4")
    parser.add_argument("--video_dir", type=Path, default=Path(__file__).resolve().parents[2] / "videos" / "grasp_stability")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    if args.lift_height <= 0 or args.hold_steps < 1:
        parser.error("lift_height and hold_steps must be positive")
    if not math.isfinite(args.observe_seconds) or args.observe_seconds <= 0:
        parser.error("observe_seconds must be positive and finite")
    if not math.isfinite(args.bend_stiffness) or args.bend_stiffness <= 0:
        parser.error("bend_stiffness must be positive and finite")
    if args.picker_backend == "attachment":
        from virtual_attachment_picker import create_picker, filter_gripper_contacts
    else:
        from a_single_point_picker import create_picker
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    material = cfg.scene.deformable.spawn.physics_material
    original_bend = material.surface_bend_stiffness
    material.surface_bend_stiffness = args.bend_stiffness
    material.youngs_modulus = 2e5 / 3
    material.surface_thickness = 0.003
    material.density = 1000.0 / 3
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.spawn.size = (0.30, 0.30)
    cfg.scene.deformable.spawn.edge_refinement = 32
    print(f"[A_single_point] surface_bend_stiffness: {original_bend:g} -> {args.bend_stiffness:g}", flush=True)
    cfg.sim.device = args.device
    cfg.seed = 0
    cfg.scene.num_envs = 1
    # update() is called once per env.step, so step at the physics frequency.
    cfg.decimation = 1
    cfg.episode_length_s = 300.0
    cfg.scene.support_neg_y = None
    cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.size = (1.5, 0.9, 1.05)
    cfg.scene.table.spawn.visible = True
    cfg.scene.deformable.init_state.pos = (0.5, -0.05, 0.012)
    cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.events.reset_deformable = None
    if args.picker_backend == "attachment":
        cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts, mode="prestartup")
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
    cfg.viewer.lookat = (0.48, -0.1, 0.03)
    cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=cfg.viewer.eye, lookat=cfg.viewer.lookat)

    video_dir = None
    completed = False
    observation_stable = False
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.kit import KitVisualizerCfg

        # Capture at about 30 Hz, preserving simulation time during playback.
        stride = max(1, round(1.0 / (30 * cfg.sim.dt)))
        fps = max(1, round(1.0 / (stride * cfg.sim.dt)))
        video_dir = args.video_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        video_dir.mkdir(parents=True, exist_ok=False)
        args.enable_cameras = True
        args.visualizer = ["kit"]
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
            eye=cfg.viewer.eye, lookat=cfg.viewer.lookat,
            window_width=640, window_height=480,
        )]
        cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:kit", output_dir=str(video_dir),
            output_filename_prefix=args.picker_backend, fps=fps, frame_stride=stride,
            video_length=sys.maxsize,
        )]
        # Store metrics alongside each recording so repeat runs remain comparable.
        args.metrics_path = video_dir / "observation.csv"
        print(f"[A_single_point] Recording {fps} FPS video to {video_dir}", flush=True)

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        gate_video_recorders(env, lambda: completed and observation_stable)
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
            towel = TowelVisual(cloth, 0.003,
                                Path(__file__).resolve().parents[2] / "assets" / "materials" / "towel")
            robot = scene["robot"]
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

            def step(capture=False):
                if picker.backend in ("kinematic", "physx_attachment"):
                    picker.update(dt=base.physics_dt)
                towel.update()
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("Episode ended during demonstration; retry with a longer episode")
                sample = None
                if capture:
                    # Read BEFORE projection: measuring only the overwritten
                    # vertex would conceal solver motion and correction jumps.
                    sample = (
                        as_tensor(cloth.data.nodal_pos_w)[0].clone(),
                        as_tensor(cloth.data.nodal_vel_w)[0].clone(),
                        anchor().clone(),
                    )
                if picker.backend == "state_projection":
                    # Enforce the virtual constraint AFTER the cloth solver.
                    # Pre-step-only writes are undone by the stiff cloth solver.
                    picker.update(dt=base.physics_dt)
                return sample

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

            def observe():
                nonlocal observation_stable
                count = max(4, math.ceil(args.observe_seconds / base.physics_dt))
                free = torch.ones(as_tensor(cloth.data.nodal_pos_w).shape[1], dtype=torch.bool, device=base.device)
                free[vertex] = False
                start_anchor = anchor().clone()
                rows = []
                last_positions = []
                print(f"[virtual_grasp] OBSERVE START: stationary target for {args.observe_seconds:.1f} simulated seconds", flush=True)
                for i in range(count):
                    positions, velocities, measured_anchor = step(capture=True)
                    speed = torch.linalg.vector_norm(velocities[free], dim=-1)
                    if i >= count * 0.75:
                        last_positions.append(positions[free].clone())
                    raw_gap = float(torch.linalg.vector_norm(positions[vertex] - picker.target_position_w))
                    values = (
                        (i + 1) * base.physics_dt,
                        float(torch.sqrt(torch.mean(speed ** 2))), float(speed.max()),
                        float(positions[free, 2].amin()), float(positions[free, 2].amax()),
                        raw_gap, float(torch.linalg.vector_norm(measured_anchor - start_anchor)),
                    )
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError("Non-finite cloth motion detected during stationary observation")
                    rows.append(values)
                    if (i + 1) % max(1, round(1.0 / base.physics_dt)) == 0:
                        print(f"[virtual_grasp] observe t={values[0]:.1f}s, free-rms-speed={values[1]:.4f} m/s, max-speed={values[2]:.4f} m/s, solver-gap={raw_gap:.4f} m", flush=True)
                args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with args.metrics_path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.writer(output)
                    writer.writerow(("time_s", "free_rms_speed_m_s", "free_max_speed_m_s",
                                     "free_min_z_m", "free_max_z_m", "pre_projection_grasp_error_m",
                                     "tcp_drift_m"))
                    writer.writerows(rows)
                quarter = max(1, len(rows) // 4)
                first = sum(row[1] for row in rows[:quarter]) / quarter
                last = sum(row[1] for row in rows[-quarter:]) / quarter
                stack = torch.stack(last_positions)
                shape_rms = float(torch.sqrt(((stack - stack.mean(0)) ** 2).sum(-1).mean()))
                last_gap = max(row[5] for row in rows[-quarter:])
                summary = {"backend": args.picker_backend,
                           "last_quarter_rms_speed_m_s": last,
                           "last_quarter_shape_rms_m": shape_rms,
                           "last_quarter_max_grasp_error_m": last_gap,
                           "stable": last < 0.02 and shape_rms < 0.002 and last_gap < 0.005,
                           "note": "Provisional numerical criteria; inspect recorded video as well"}
                args.metrics_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                observation_stable = summary["stable"]
                print(f"[stability] {summary}", flush=True)
                print(f"[virtual_grasp] OBSERVE END: first-quarter RMS={first:.4f} m/s, last-quarter RMS={last:.4f} m/s; metrics={args.metrics_path.resolve()}", flush=True)

            def move(goal_w, count=240, goal_quat_w=None):
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
                for _ in range(1200):
                    step()
                    if float(torch.linalg.vector_norm(anchor() - goal_w)) < 0.005:
                        break
                error = float(torch.linalg.vector_norm(anchor() - goal_w))
                print(f"[virtual_grasp] move goal={goal_w.tolist()}, actual={anchor().tolist()}, error={error:.4f} m", flush=True)
                if error >= 0.005:
                    raise RuntimeError(f"IK did not reach target: {error:.4f} m; goal={goal_w.tolist()}, actual={anchor().tolist()}")

            with torch.inference_mode():
                hold(120)
                print(f"[virtual_grasp] initial TCP={anchor().tolist()}, quat={target_quat_w.tolist()}", flush=True)
                nodes = as_tensor(cloth.data.nodal_pos_w)[0]
                # Pick the nearest vertex to the x-min/y-min corner after settling.
                corner_xy = nodes[:, :2].amin(dim=0)
                vertex = int(((nodes[:, :2] - corner_xy) ** 2).sum(dim=-1).argmin())
                initial = nodes[vertex].clone()
                print(f"[virtual_grasp] backend={picker.backend}, vertex={vertex}")
                above = initial.clone()
                above[2] += 0.07
                # A fixed downward grasp pose avoids preserving a randomly tilted
                # wrist that cannot reach the table without collision/joint limits.
                move(above, goal_quat_w=action.new_tensor((1.0, 0.0, 0.0, 0.0)))
                # Re-read the moving cloth before descending with open fingers.
                touch = as_tensor(cloth.data.nodal_pos_w)[0, vertex].clone()
                touch[2] += 0.015
                move(touch, 180)
                set_gripper(closed=True)
                current = as_tensor(cloth.data.nodal_pos_w)[0, vertex].clone()
                if not picker.grasp(current, vertex_id=vertex):
                    distance = float(torch.linalg.vector_norm(current - anchor()))
                    raise RuntimeError(f"Gripper too far from corner: {distance:.4f} m; check IK pose")
                attached = current.clone()
                lift = anchor().clone()
                lift[2] += args.lift_height
                move(lift, 360)
                observe()
                raised = as_tensor(cloth.data.nodal_pos_w)[0, vertex].clone()
                rise = float(raised[2] - attached[2])
                error = float(torch.linalg.vector_norm(raised - picker.target_position_w))
                print(f"[virtual_grasp] lift-rise={rise:.4f} m, follow-error={error:.4f} m")
                if rise < args.lift_height * 0.5 or error > 0.03:
                    raise RuntimeError("Lift/follow check failed")
                shifted = lift.clone()
                if args.move_distance != 0.0:
                    shifted[1] += args.move_distance
                    move(shifted)
                lowered = shifted.clone()
                lowered[2] -= args.lift_height
                move(lowered, 360)
                set_gripper(closed=False)
                picker.release()
                released = as_tensor(cloth.data.nodal_pos_w)[0, vertex].clone()
                retreat = anchor().clone()
                retreat[2] += 0.08
                move(retreat)
                hold(args.hold_steps)
                final = as_tensor(cloth.data.nodal_pos_w)[0, vertex]
                # A released node must not follow an 8 cm gripper retreat.
                if float(final[2] - released[2]) > 0.04:
                    raise RuntimeError("Release check failed: cloth rose during gripper retreat")
                print("[virtual_grasp] PASS: gripper close/open, lift, follow, release and retreat checks passed; vibration metrics are reported separately")
                completed = True
        finally:
            if picker is not None:
                picker.reset()
            env.close()
            if video_dir is not None and completed and observation_stable:
                clips = list(video_dir.glob("*.mp4"))
                if not clips or not all(clip.stat().st_size > 0 for clip in clips):
                    raise RuntimeError(f"No video was saved; check the recording errors above: {video_dir}")
                for clip in clips:
                    print(f"[A_single_point] Video saved: {clip}", flush=True)
            elif video_dir is not None:
                print("[stability] Unsuccessful trial: video frames discarded without saving MP4.", flush=True)


if __name__ == "__main__":
    main()
