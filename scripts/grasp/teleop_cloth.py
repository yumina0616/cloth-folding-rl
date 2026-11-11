"""Keyboard cloth folding with the shared PR #27 demonstration format."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.visualizers import VisualizerCfg
import isaaclab.sim as sim_utils
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab_tasks  # noqa: F401

from a_single_point_picker import as_tensor
from virtual_attachment_picker import filter_gripper_contacts
from physical_picker import create_picker, pinch_vertex
from rectangular_fold_geometry import rectangular_pairs, pair_distances
from towel_visual import TowelVisual
from scripts.corner_tracking.cloth_corner_tracker import detect_corner_vertex_indices
from scripts.learning.teleop_recorder import TeleopRecorder


def reset_cloth(env, env_ids):
    """Restore the flat initial mesh on every explicit reset."""
    cloth = env.scene["deformable"]
    state = as_tensor(cloth.data.default_nodal_state_w)[env_ids].clone()
    cloth.write_nodal_state_to_sim_index(state, env_ids=env_ids)


def configure(args):
    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.decimation = 1  # Refresh the PhysX attachment every physics step.
    cfg.sim.render_interval = max(1, round(1 / (60 * cfg.sim.dt)))
    cfg.scene.support_neg_y = cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.size = (1.5, 0.9, 1.05)
    cfg.scene.table.spawn.visible = True
    cfg.scene.deformable.spawn.size = (0.30, 0.30)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    cfg.scene.deformable.spawn.deformable_props.self_collision = True
    cfg.scene.deformable.init_state.pos = (0.50, -0.05, 0.012)
    cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    material = cfg.scene.deformable.spawn.physics_material
    material.surface_thickness = 0.003
    material.density = 1000 / 3
    material.youngs_modulus = 2e5 / 3
    material.surface_bend_stiffness = 2e4 / 108
    cfg.events.reset_deformable = EventTermCfg(func=reset_cloth, mode="reset")
    if args.grasp_mode == "virtual":
        cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts, mode="prestartup")
    else:
        cfg.scene.robot.spawn.make_uninstanceable = True
        cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=1.5, restitution=0., friction_combine_mode="max")
        hand = cfg.scene.robot.actuators["panda_hand"]
        hand.joint_effort_limit, hand.stiffness, hand.damping = 70., 500., 100.
    cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
    cfg.curriculum.gravity = cfg.curriculum.action_rate = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    cfg.actions = type(cfg)().actions.ik
    if args.grasp_mode == "physical":
        cfg.actions.gripper_action.open_command_expr = {"panda_finger_joint1": args.open_width / 2}
    cfg.actions.arm_action.body_offset.pos = (0.0, 0.0, 0.1034)
    cfg.actions.arm_action.controller.ik_params = {"lambda_val": 0.1}
    # Manual resets only: automatic reset must never occur with an attachment active.
    for name in vars(cfg.terminations):
        setattr(cfg.terminations, name, None)
    cfg.viewer.eye = (1.15, 0.60, 0.60)
    cfg.viewer.lookat = (0.50, -0.05, 0.03)
    cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=cfg.viewer.eye, lookat=cfg.viewer.lookat)
    return cfg


class KeyboardInput:
    """Use Kit's input events; repeats cannot accumulate motion or toggle twice."""

    def __init__(self):
        import carb.input
        import omni.appwindow

        self.carb = carb.input
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.held = set()
        self.events = deque()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_event)

    def on_event(self, event, *unused):
        # CHAR events carry text, whereas physical key events may carry an enum
        # or a string depending on the Kit binding. Ignore text and OS repeats.
        if event.type not in (self.carb.KeyboardEventType.KEY_PRESS,
                              self.carb.KeyboardEventType.KEY_RELEASE):
            return True
        key = getattr(event.input, "name", event.input)
        if not isinstance(key, str):
            return True
        key = key.upper()
        if event.type == self.carb.KeyboardEventType.KEY_PRESS:
            if key not in self.held and key in ("K", "P", "R", "ESCAPE"):
                self.events.append(key)
            self.held.add(key)
        elif event.type == self.carb.KeyboardEventType.KEY_RELEASE:
            self.held.discard(key)
        return True

    def direction(self):
        return [int(plus in self.held) - int(minus in self.held)
                for plus, minus in (("W", "S"), ("A", "D"), ("Q", "E"))]

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)


class ClothTeleop:
    def __init__(self, env, args):
        self.env, self.base, self.args = env, env.unwrapped, args
        self.cloth = self.base.scene["deformable"]
        self.robot = self.base.scene["robot"]
        self.ee = self.base.scene["ee_frame"]
        self.dt = self.base.physics_dt
        self.picker = create_picker(self.cloth, self.tcp, grasp_mode=args.grasp_mode, grasp_radius=args.grasp_radius)
        self.towel = TowelVisual(self.cloth, 0.003, ROOT / "assets/materials/towel")
        self.recorder = None
        self.reset()

    def tcp(self):
        return as_tensor(self.ee.data.target_pos_w)[0, 0]

    def corners(self):
        return as_tensor(self.cloth.data.nodal_pos_w)[0, self.corner_ids]

    def reset(self):
        self.picker.reset()  # Always detach BEFORE resetting physics.
        if self.recorder is not None and not self.finished:
            self.recorder.finish(self.corners(), self.episode_id, aborted=True)
        self.env.reset()
        self.base.command_manager.get_term("deformable_pose").success_visualizer.set_visibility(False)
        self.action = torch.zeros(self.base.action_space.shape, device=self.base.device)
        if self.action.shape != (1, 8):
            raise RuntimeError(f"Expected absolute pose IK (1,8), got {self.action.shape}")
        self.target = self.tcp().clone()
        # Preserve current wrist pose during settling; rotate gradually to top-down later.
        self.orientation = as_tensor(self.ee.data.target_quat_w)[0, 0].clone()
        self.action[0, 7] = 1
        self.steps = 0
        self.finished = False
        self.mode = "initialization"
        self.recorder = None
        for _ in range(max(1, round(self.args.settle_seconds / self.dt))):
            self.step(record=False)
        self.corner_ids = detect_corner_vertex_indices(as_tensor(self.cloth.data.nodal_pos_w)[0])
        nodes = as_tensor(self.cloth.data.nodal_pos_w)[0]
        self.pinch_ids = [pinch_vertex(nodes[:, :2], int(vertex), self.corners()[:, :2].mean(0),
                                      self.args.open_width) for vertex in self.corner_ids]
        self.pairs = rectangular_pairs(self.corners(), as_tensor(self.robot.data.root_pos_w)[0, :2])
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.episode_dir = self.args.output_dir.resolve() / self.episode_id
        self.recorder = TeleopRecorder(self.episode_dir, self.pairs, self.args.success_distance)
        # Align wrist while holding the measured TCP at a safe height.
        start = self.orientation.clone()
        end = self.action.new_tensor([1., 0., 0., 0.])
        if float(torch.dot(start, end)) < 0:
            end = -end
        for index in range(1, 121):
            self.orientation = torch.nn.functional.normalize(torch.lerp(start, end, index / 120), dim=0)
            self.step(record=False)
        self.steps = 0
        self.mode = "ready"
        self.deadline = 0
        self.place_xy = None
        self.completed_actions = 0
        self.grasp_verified = False
        self.grasp_checks = []
        print(f"[teleop] READY {self.episode_id}; pairs={self.pairs}; corners={self.corners().tolist()}", flush=True)
        if self.args.grasp_mode == "physical":
            print(f"[teleop] PHYSICAL: move to inward pinch points={nodes[self.pinch_ids].tolist()}; "
                  "K only opens/closes fingers; move and lift manually", flush=True)

    def step(self, *, record=True):
        pos, quat = subtract_frame_transforms(
            as_tensor(self.robot.data.root_pos_w), as_tensor(self.robot.data.root_quat_w),
            self.target.unsqueeze(0), self.orientation.unsqueeze(0),
        )
        self.action[0, :3], self.action[0, 3:7] = pos[0], quat[0]
        self.picker.update(dt=self.dt)
        _, _, terminated, truncated, _ = self.env.step(self.action)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("Unexpected environment reset during teleoperation")
        self.towel.update()
        self.steps += 1
        if record and not self.finished:
            if self.args.grasp_mode == "physical":
                self.observe_physical_grasp()
            self.recorder.append_step(
                self.steps * self.dt, self.mode, self.action[0],
                as_tensor(self.robot.data.joint_pos)[0], self.corners(), self.tcp(),
                self.picker.is_grasping and (self.args.grasp_mode == "virtual" or self.grasp_verified),
            )

    def toggle_grasp(self):
        if self.finished or self.mode not in ("ready", "carry"):
            return
        if self.args.grasp_mode == "physical":
            self.toggle_physical_gripper()
            return
        if self.picker.is_grasping:
            self.place_xy = self.tcp()[:2].detach().cpu().tolist()
            self.action[0, 7] = 1
            self.mode = "opening"
            self.deadline = self.steps + max(1, round(0.25 / self.dt))
            return
        current = self.corners().clone()
        candidates = current
        slot = int(torch.linalg.vector_norm(candidates - self.tcp(), dim=-1).argmin())
        vertex = int(self.corner_ids[slot])
        if not self.picker.grasp(candidates[slot], vertex_id=vertex):
            distance = float(torch.linalg.vector_norm(candidates[slot] - self.tcp()))
            print(f"[teleop] GRASP MISSED: corner {slot}, distance={distance:.3f}m; move closer and press K", flush=True)
            return
        self.recorder.begin_pick_place(current, candidates[slot, :2])
        self.action[0, 7] = -1
        self.mode = "carry"
        self.grasp_verified = False
        print(f"[teleop] GRASP corner={slot}, vertex={vertex}", flush=True)

    def toggle_physical_gripper(self):
        """Command only the fingers. Proximity/monitoring never vetoes a command."""
        if float(self.action[0, 7]) < 0:
            self.place_xy = self.tcp()[:2].detach().cpu().tolist()
            self.action[0, 7] = 1
            self.picker.release()
            self.grasp_verified = False
            # Close the segment after the next regular sample of the open action.
            # Movement remains enabled during this one-step recording boundary.
            self.mode = "physical_release"
            print("[teleop] OPEN (manual)", flush=True)
            return
        self.recorder.begin_pick_place(self.corners(), self.tcp()[:2])
        self.action[0, 7] = -1
        self.mode = "carry"
        self.grasp_verified = False
        nodes = as_tensor(self.cloth.data.nodal_pos_w)[0]
        vertex = int(torch.linalg.vector_norm(nodes - self.tcp(), dim=-1).argmin())
        # This picker only observes a material point, never attaches or moves it.
        if self.picker.grasp(nodes[vertex], vertex_id=vertex):
            self.check_start_z = float(nodes[vertex, 2])
            self.check_tcp_z = float(self.tcp()[2])
        print("[teleop] CLOSE (manual); grasp state is an estimate only", flush=True)

    def observe_physical_grasp(self):
        """Label observations without changing targets, fingers, or save eligibility."""
        estimate = False
        rise = hand_rise = drift = None
        if self.picker.is_grasping and float(self.action[0, 7]) < 0:
            rise = float(self.picker.target_position_w[2]) - self.check_start_z
            hand_rise = float(self.tcp()[2]) - self.check_tcp_z
            drift = self.picker.drift()
            estimate = rise >= .015 and hand_rise >= .025 and drift < .035
        if estimate != self.grasp_verified:
            self.grasp_checks.append({"time_s": self.steps * self.dt, "rise_m": rise,
                                      "hand_rise_m": hand_rise, "drift_m": drift, "estimated_grasping": estimate})
            print(f"[teleop] Grasp estimate={estimate} (recording only)", flush=True)
        self.grasp_verified = estimate

    def request_finish(self):
        if self.finished:
            return
        if self.mode != "ready" or self.picker.is_grasping or not self.completed_actions:
            print("[teleop] Finish requires at least one completed pick/place and released cloth", flush=True)
            return
        self.mode = "final_settle"
        self.deadline = self.steps + max(1, round(self.args.settle_seconds / self.dt))
        print("[teleop] Holding still for final success check...", flush=True)

    def tick(self, direction=(0, 0, 0)):
        if self.finished:
            self.step(record=False)
            return
        if self.mode in ("ready", "carry", "physical_release"):
            delta = self.target.new_tensor(direction)
            delta /= torch.linalg.vector_norm(delta).clamp_min(1)
            # Avoid accumulating unreachable IK targets when the arm cannot follow.
            candidate = self.target + delta * self.args.speed * self.dt
            candidate = torch.maximum(torch.minimum(candidate, self.target.new_tensor([.78, .42, .60])),
                                      self.target.new_tensor([.15, -.42, .004]))
            error = float(torch.linalg.vector_norm(candidate - self.tcp()))
            previous_error = float(torch.linalg.vector_norm(self.target - self.tcp()))
            if error < 0.08 or error < previous_error:
                self.target = candidate
        self.step()
        if self.mode == "physical_release":
            self.recorder.end_pick_place(self.corners(), self.place_xy)
            self.completed_actions += 1
            self.mode = "ready"
        elif self.mode == "opening" and self.steps >= self.deadline:
            self.picker.release()
            self.grasp_verified = False
            self.mode = "release_settle"
            self.deadline = self.steps + max(1, round(self.args.settle_seconds / self.dt))
        elif self.mode == "release_settle" and self.steps >= self.deadline:
            self.recorder.end_pick_place(self.corners(), self.place_xy)
            self.completed_actions += 1
            self.mode = "ready"
            print(f"[teleop] RELEASED; pair distances={pair_distances(self.corners(), self.pairs)}", flush=True)
        elif self.mode == "final_settle" and self.steps >= self.deadline:
            saved = self.recorder.finish(self.corners(), self.episode_id)
            self.finished = True
            if saved:
                self.write_context()
            print(f"[teleop] {'SAVED ' + str(self.episode_dir) if saved else 'FAILED: no dataset saved'}; "
                  f"pair distances={pair_distances(self.corners(), self.pairs)}; press R for next episode", flush=True)
        if not self.finished and self.steps * self.dt >= self.args.max_episode_seconds:
            print("[teleop] Episode time limit reached; discarding and resetting", flush=True)
            self.reset()

    def write_context(self):
        # Keep the three PR #27 files unchanged. Document conventions alongside them.
        context = {
            "source": "keyboard_teleoperation", "grasp_mode": self.args.grasp_mode,
            "action_type": "absolute_pose_ik", "action_frame": "robot_root",
            "quaternion_order": "wxyz", "position_unit": "metre", "time_unit": "second",
            "gripper_open": 1, "gripper_close": -1,
            "corner_vertex_indices": self.corner_ids.tolist(), "pairs": self.pairs,
            "sample_dt": self.dt, "success_threshold_m": self.args.success_distance,
            "final_pair_distances_m": pair_distances(self.corners(), self.pairs),
            "pick_xy": "measured TCP at manual close" if self.args.grasp_mode == "physical" else "measured cloth corner at grasp",
            "grasping_semantics": "passive position-based estimate" if self.args.grasp_mode == "physical" else "virtual attachment active",
            "place_xy": "measured TCP immediately before opening", "physical_grasp_checks": self.grasp_checks,
            "config": vars(self.args),
        }
        (self.episode_dir / "context.json").write_text(json.dumps(context, default=str, indent=2), encoding="utf-8")

    def close(self):
        self.picker.reset()
        if self.recorder is not None and not self.finished:
            self.recorder.finish(self.corners(), self.episode_id, aborted=True)


def smoke_test(session):
    """Exercise real physics without saving a fake successful demonstration."""
    def move(goal, seconds=2):
        start = session.target.clone()
        count = max(1, round(seconds / session.dt))
        for step in range(1, count + 1):
            session.target = torch.lerp(start, goal, step / count)
            session.tick()

    slot = session.pairs[0][0]
    goal = (as_tensor(session.cloth.data.nodal_pos_w)[0, session.pinch_ids[slot]].clone()
            if session.args.grasp_mode == "physical" else session.corners()[slot].clone())
    hover = goal.clone()
    hover[2] += .08
    move(hover)
    move(goal, 1)
    initial_z = float(session.corners()[slot, 2])
    session.toggle_grasp()
    if session.args.grasp_mode == "physical":
        held_target = session.target.clone()
        for _ in range(max(1, round(.75 / session.dt))):
            session.tick()
        if not torch.allclose(session.target, held_target, atol=.003):
            raise AssertionError("Closing the fingers moved the arm automatically")
    if session.args.grasp_mode == "virtual" and not session.picker.is_grasping:
        raise AssertionError("Smoke test failed to grasp")
    goal = session.target.clone()
    goal[2] += .06
    move(goal, 1)
    rise = float(session.corners()[slot, 2]) - initial_z
    if rise < .02 and session.args.grasp_mode == "virtual":
        raise AssertionError(f"Attached cloth did not rise: {rise}")
    session.toggle_grasp()
    while session.mode != "ready":
        session.tick()
    if session.picker.is_grasping or session.completed_actions != 1:
        raise AssertionError("Release or segment completion failed")
    session.request_finish()
    while not session.finished:
        session.tick()
    if session.episode_dir.exists():
        raise AssertionError("Unfolded smoke test should not save a successful dataset")
    session.reset()
    if session.picker.is_grasping or session.completed_actions or session.recorder.rows:
        raise AssertionError("Reset did not clear the episode")
    distances = pair_distances(session.corners(), session.pairs)
    if min(distances) < .20:
        raise AssertionError(f"Reset did not restore the flat cloth: {distances}")
    if session.args.grasp_mode == "physical":
        print(f"[teleop] PHYSICAL SMOKE PASS: manual close/lift/open, failed-save gate, reset; "
              f"measured corner rise={rise:.3f}m (no automatic grasp check)", flush=True)
    else:
        print(f"[teleop] SMOKE PASS: grasp, lift ({rise:.3f}m), release, failed-save gate, flat-mesh reset", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_mode", choices=["virtual", "physical"], default="virtual")
    parser.add_argument("--open_width", type=float, default=.04)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=.06, help="TCP translation speed in m/s of simulation time")
    parser.add_argument("--grasp_radius", type=float, default=.025)
    parser.add_argument("--success_distance", type=float, default=.02)
    parser.add_argument("--settle_seconds", type=float, default=2.)
    parser.add_argument("--max_episode_seconds", type=float, default=600.)
    parser.add_argument("--mesh_refinement", type=float, default=32.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true", help="Finite integration test without keyboard or saved successes")
    add_launcher_args(parser)
    if "--smoke_test" not in sys.argv:
        parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    if args.output_dir is None:
        args.output_dir = ROOT / "runs/teleop" / args.grasp_mode
    if not math.isfinite(args.open_width) or not 0 < args.open_width <= .08:
        parser.error("open_width must be in (0, 0.08] metres")
    for key in ("speed", "grasp_radius", "success_distance", "settle_seconds", "max_episode_seconds", "mesh_refinement"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive and finite")
    args.require_kit = True
    sys.argv = [sys.argv[0], *hydra_args]
    cfg = configure(args)
    with launch_simulation(cfg, args):
        import omni.kit.app

        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        session = keyboard = None
        try:
            with torch.inference_mode():
                session = ClothTeleop(env, args)
                if args.smoke_test:
                    smoke_test(session)
                    return
                keyboard = KeyboardInput()
                print("[teleop] W/S=X A/D=Y Q/E=Z | K=grasp/release P=finish R=discard/reset ESC=quit", flush=True)
                running = True
                while running and omni.kit.app.get_app().is_running():
                    started = time.perf_counter()
                    while keyboard.events:
                        event = keyboard.events.popleft()
                        if event == "ESCAPE":
                            running = False
                            break
                        if event == "R":
                            keyboard.held.clear()
                            keyboard.events.clear()
                            session.reset()
                        elif event == "K":
                            session.toggle_grasp()
                        elif event == "P":
                            session.request_finish()
                    if not running:
                        break
                    session.tick(keyboard.direction())
                    time.sleep(max(0, session.dt - (time.perf_counter() - started)))
        except KeyboardInterrupt:
            print("[teleop] Interrupted; unsaved episode discarded", flush=True)
        finally:
            if keyboard is not None:
                keyboard.close()
            try:
                if session is not None:
                    session.close()
            finally:
                env.close()


if __name__ == "__main__":
    main()
