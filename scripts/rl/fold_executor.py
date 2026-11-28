"""Run "grasp a corner" and "place the held corner at (x, y)" in N environments at once.

Needs Isaac Sim (run through the Isaac Lab launcher). grasp_mode = "physical" (default) or "virtual".

Why every environment moves in lockstep: Isaac Lab steps all N environments in one physics
scene, and env.step() takes one action row per environment. So each call below runs the same
number of physics steps in every environment. Environments that are not taking part
(active = False) keep their hand where it is and just wait.

Motion and numbers follow scripts/grasp/rectangular_fold_demo.py (PR #28 version for the physical grasp):
    physical: approach 8 cm above the pinch point (just inside the corner, chosen on the flat cloth),
              descend with the fingertips on the cloth, close, grasp check (slow 6 cm lift and hold;
              the corner must rise >= 15 mm), one retry, carry with the LIVE corner-to-hand offset,
              watch slip and drop while carrying.
    virtual:  descend to 1.5 cm above the corner, close, attach the corner vertex (no check).
Differences from the demo: the place point is the (x, y) given by the policy, fixed at decision time
(the demo follows the destination corner, and its physical default "mirror" placement needs a
destination corner, so it is not used); a miss or a failed grasp is returned, not raised.

A hybrid-policy caller can additionally pass pick_xy to grasp_corner() and send
(N,8) absolute root-frame actions to step_policy_action(). The latter is an
execution interface, not a complete PPO environment. Set policy_controls_lift=True to stop after closing and let the policy
lift, carry and release. finish_policy_place() can assess a completed release
for integration tests, but is not yet a PPO-compatible Gym environment.

Typical use (see run_executor_smoke.py):
    configure_fold_cfg(cfg, num_envs, grasp_mode="physical")   # before gym.make
    executor = BatchedFoldExecutor(env, grasp_mode="physical")  # after env.reset()
    executor.detect_corners()                                   # once, on the flat cloth
    ok, info = executor.grasp_corner(source)                    # source: (N,) corner number 0-3
    miss, info = executor.place_held_corner(place_xy, active=ok)  # place_xy: (N, 2) where the corner lands
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from isaaclab.utils.math import subtract_frame_transforms

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "scripts" / "grasp", ROOT / "scripts" / "corner_tracking"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cloth_corner_tracker import detect_corner_vertex_indices  # noqa: E402
from physical_picker import as_tensor, create_picker, pinch_vertex  # noqa: E402

from scripts.rl.fold_mdp import cloth_dropped, fold_success, pair_distances  # noqa: E402
from scripts.rl.fold_motion import (  # noqa: E402
    APPROACH_HEIGHT, RETREAT_HEIGHT, TOUCH_HEIGHT,
    carry_arc_corner_points, lower_goal, place_point_from_xy, smoothstep_path,
)

REACH_TOLERANCE = 0.005           # m, same as the demo's move()
DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)  # world hand orientation for the approach (demo, hand yaw 0)
GRASP_MODES = ("physical", "virtual")


def filter_gripper_contacts_all_envs(env, env_ids=None):
    """Virtual grasp: stop hand/finger shapes from colliding with the cloth, in EVERY environment.

    Same as virtual_attachment_picker.filter_gripper_contacts, which only handles environment 0.
    Must run before physics starts (EventTermCfg mode="prestartup").
    """
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage
    hand_names = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    for root in env.scene.env_prim_paths:
        robot_root = stage.GetPrimAtPath(root + "/Robot")
        hand_paths = [prim.GetPath() for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
                      if prim.HasAPI(UsdPhysics.CollisionAPI)
                      and any(name in str(prim.GetPath()) for name in hand_names)]
        cloth_root = stage.GetPrimAtPath(root + "/Deformable")
        colliders = [prim for prim in Usd.PrimRange(cloth_root) if prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not hand_paths or not colliders:
            raise RuntimeError(f"Missing hand or cloth collision shapes under {root}")
        for prim in colliders:
            targets = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            for path in hand_paths:
                targets.AddTarget(path)
    print(f"[executor] filtered gripper-cloth contacts in {len(env.scene.env_prim_paths)} environments", flush=True)


def configure_fold_cfg(cfg, num_envs, *, grasp_mode="physical", open_width=0.04, finger_friction=1.5,
                       finger_static_friction=2.0, gripper_effort=70.0):
    """Task settings the fold needs, copied from rectangular_fold_demo.py. Cloth/table material is left to the caller."""
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg

    if grasp_mode not in GRASP_MODES:
        raise ValueError(f"grasp_mode must be one of {GRASP_MODES}")
    cfg.scene.num_envs = num_envs
    cfg.decimation = 1             # one env.step = one physics step
    cfg.episode_length_s = 300.0   # long enough that time_out never cuts a fold short
    cfg.scene.support_neg_y = None
    cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.visible = True
    cfg.events.reset_deformable = None
    if grasp_mode == "virtual":
        # The attachment holds the cloth, so the fingers must not push it
        cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts_all_envs, mode="prestartup")
    else:
        # Finger contact and friction hold the cloth (008 gripper settings, demo PR #28)
        cfg.scene.robot.spawn.make_uninstanceable = True
        cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=finger_static_friction, dynamic_friction=finger_friction,
            restitution=0.0, friction_combine_mode="max")
        cfg.scene.robot.actuators["panda_hand"].joint_effort_limit = gripper_effort
        cfg.scene.robot.actuators["panda_hand"].stiffness = 500.0
        cfg.scene.robot.actuators["panda_hand"].damping = 100.0
    cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
    cfg.curriculum.gravity = None
    cfg.curriculum.action_rate = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    cfg.actions = type(cfg)().actions.ik
    if grasp_mode == "physical":
        cfg.actions.gripper_action.open_command_expr = {"panda_finger_joint1": open_width / 2}  # one joint = one finger
    cfg.actions.arm_action.body_offset.pos = (0.0, 0.0, 0.1034)  # measured TCP used by ee_frame
    cfg.actions.arm_action.controller.ik_params = {"lambda_val": 0.1}
    for name in vars(cfg.terminations):
        if name != "time_out":
            setattr(cfg.terminations, name, None)
    return cfg


class BatchedFoldExecutor:
    def __init__(self, env, *, grasp_mode="physical", table_z=0.0, cloth_thickness=0.003, lift_height=0.12,
                 settle_seconds=3.0, grasp_radius=None,
                 # physical grasp (demo defaults, experiment 008 / PR #28)
                 open_width=0.04, pad_half=0.01, edge_margin=0.005, press_depth=0.0,
                 grasp_settle_s=0.5, check_lift=0.06, check_lift_s=2.0, check_hold_s=1.0,
                 check_min_rise=0.015, quality_max=0.0, retries=1,
                 slip_drift=0.02, drop_height=0.01, drop_hand_height=0.06):
        if grasp_mode not in GRASP_MODES:
            raise ValueError(f"grasp_mode must be one of {GRASP_MODES}")
        self.env = env
        self.base = env.unwrapped
        scene = self.base.scene
        self.cloth, self.ee, self.robot = scene["deformable"], scene["ee_frame"], scene["robot"]
        self.num_envs = self.base.num_envs
        self.device = self.base.device
        self.physical = grasp_mode == "physical"
        self.grasp_mode = grasp_mode
        self.table_z, self.cloth_thickness, self.lift_height = table_z, cloth_thickness, lift_height
        dt = self.base.physics_dt
        self.settle_steps = math.ceil(settle_seconds / dt)
        self.open_width, self.pad_half, self.edge_margin, self.press_depth = open_width, pad_half, edge_margin, press_depth
        self.grasp_settle_steps = math.ceil(grasp_settle_s / dt)
        self.check_lift, self.check_lift_steps = check_lift, max(1, math.ceil(check_lift_s / dt))
        self.check_hold_steps = math.ceil(check_hold_s / dt)
        self.check_min_rise, self.quality_max = check_min_rise, quality_max
        self.retries = retries if self.physical else 0
        self.slip_drift, self.drop_height, self.drop_hand_height = slip_drift, drop_height, drop_hand_height

        self.action = torch.zeros(self.base.action_space.shape, device=self.device)
        if self.action.shape != (self.num_envs, 8):
            raise RuntimeError(f"Expected absolute pose IK action (N, 8), got {tuple(self.action.shape)}")
        self.target_w = self.tcp().clone()
        self.target_quat_w = as_tensor(self.ee.data.target_quat_w)[:, 0].clone()
        self.action[:, 7] = 1.0  # gripper open
        self._write_pose()

        # One picker per environment. Physical: tracks the held corner (6 cm reach, the pinch point is
        # about 4 cm from the corner). Virtual: attaches the corner vertex to the hand.
        radius = grasp_radius if grasp_radius is not None else (0.06 if self.physical else 0.03)
        self.pickers = [create_picker(self.cloth, (lambda i=i: self.tcp()[i]), grasp_mode=grasp_mode,
                                      env_id=i, grasp_radius=radius) for i in range(self.num_envs)]
        n = self.num_envs
        self.offset = torch.zeros(n, 3, device=self.device)       # held corner - TCP (virtual: fixed)
        self.grasp_offset = torch.zeros(n, 3, device=self.device) # offset right after the grasp, for slip
        self.held_corner = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.held_vertex = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.check_rise = torch.zeros(n, device=self.device)      # how much the grasp check lifted the corner
        self.corner_indices = None   # (N, 4) mesh vertex of each corner, found on the flat cloth
        self.pinch_indices = None    # (N, 4) physical: vertex just inside each corner where the fingers close
        # Transport monitoring (physical), updated every physics step while carrying
        self._carrying = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.max_drift = torch.zeros(n, device=self.device)
        # Low-level policy attempt tracking (all tensors are per environment).
        self.policy_attempt_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_released = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_lift_verified = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_slipped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_grip_start_z = torch.zeros(n, device=self.device)
        self.policy_release_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.policy_step_index = 0

    # ---------- reading state ----------
    def nodes(self):
        return as_tensor(self.cloth.data.nodal_pos_w)  # (N, V, 3)

    def tcp(self):
        return as_tensor(self.ee.data.target_pos_w)[:, 0]  # (N, 3)

    def _vertices(self, index):
        """Positions of one vertex per environment: index (N,) -> (N, 3)."""
        return torch.gather(self.nodes(), 1, index.view(-1, 1, 1).expand(-1, 1, 3))[:, 0]

    def detect_corners(self):
        """Call once on the flat cloth (after env.reset() and a short hold)."""
        nodes = self.nodes()
        self.corner_indices = torch.stack([detect_corner_vertex_indices(nodes[i]) for i in range(self.num_envs)])
        if self.physical:
            pinch = []
            for i in range(self.num_envs):
                flat_xy = nodes[i, :, :2].clone()
                center_xy = nodes[i, self.corner_indices[i], :2].mean(dim=0)
                pinch.append([pinch_vertex(flat_xy, int(v), center_xy, self.open_width, pad_half=self.pad_half,
                                           edge_margin=self.edge_margin, jaw_axis="y")
                              for v in self.corner_indices[i].tolist()])
            self.pinch_indices = torch.tensor(pinch, dtype=torch.long, device=self.device)
        return self.corner_indices

    def corners(self):
        """(N, 4, 3) world positions of the four tracked corners."""
        return torch.gather(self.nodes(), 1, self.corner_indices.unsqueeze(-1).expand(-1, -1, 3))

    def grasp_target(self, source):
        """Where the TCP goes down: the pinch vertex just inside the corner (physical) or the corner (virtual)."""
        rows = torch.arange(self.num_envs, device=self.device)
        table = self.pinch_indices if self.physical else self.corner_indices
        return self._vertices(table[rows, source]).clone()

    def policy_grasp_target(self, pick_xy, max_distance=0.03):
        """Find cloth mesh vertex nearest to a policy-selected world XY point.

        Returns (target_w [N,3], vertex_ids [N]). Reject XY positions farther
        than max_distance from the cloth mesh, rather than silently snapping.
        """
        nodes = self.nodes()
        pick_xy = torch.as_tensor(pick_xy, device=self.device, dtype=nodes.dtype)
        if pick_xy.shape != (self.num_envs, 2):
            raise ValueError(f"Expected pick_xy shape ({self.num_envs}, 2)")
        if not bool(torch.isfinite(pick_xy).all()):
            raise ValueError("pick_xy contains NaN or Inf")
        distances = torch.linalg.vector_norm(nodes[:, :, :2] - pick_xy[:, None, :], dim=-1)
        nearest_dist, vertex_ids = distances.min(dim=1)
        if bool((nearest_dist > max_distance).any()):
            raise ValueError("pick_xy is too far from the cloth mesh")
        return self._vertices(vertex_ids).clone(), vertex_ids

    def held_corner_position(self):
        """(N, 3) current position of the corner each environment holds (garbage where nothing is held)."""
        rows = torch.arange(self.num_envs, device=self.device)
        return self._vertices(self.corner_indices[rows, self.held_corner.clamp(min=0)])

    def held_grasp_point_position(self):
        """Tracked grip vertex, which can differ from the logical cloth corner."""
        return self._vertices(self.held_vertex.clamp(min=0))

    # ---------- low-level motion (all environments together) ----------
    def _write_pose(self, position_w=None, quat_w=None):
        position_w = self.target_w if position_w is None else position_w
        quat_w = self.target_quat_w if quat_w is None else quat_w
        # Absolute IK actions are expressed in the robot root frame
        pos_b, quat_b = subtract_frame_transforms(
            as_tensor(self.robot.data.root_pos_w), as_tensor(self.robot.data.root_quat_w), position_w, quat_w)
        self.action[:, :3] = pos_b
        self.action[:, 3:7] = quat_b

    def step(self):
        for picker in self.pickers:
            if picker.backend in ("kinematic", "physx_attachment"):
                picker.update(dt=self.base.physics_dt)  # before the physics step
        _, _, terminated, truncated, _ = self.env.step(self.action)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("An environment ended during a pick-and-place; increase episode_length_s")
        for picker in self.pickers:
            if picker.backend == "state_projection":
                picker.update(dt=self.base.physics_dt)  # after the cloth solver
        if self.physical and bool(self._carrying.any()):
            self._watch_transport()

    def step_policy_action(self, policy_action, active=None, max_step_m=0.01):
        """Execute one policy (N, 8) root-frame absolute IK action.

        A change from closed to open on a held, active environment marks a
        policy-controlled release. Physical grip/lift/slip are monitored as
        observations, not guaranteed by the finger-close command.

        Returns a dictionary of (N,) status masks; it is NOT a Gym PPO step.
        """
        mask = self._mask(active)
        action = torch.as_tensor(policy_action, device=self.device, dtype=self.action.dtype).clone()
        if action.shape != (self.num_envs, 8):
            raise ValueError(f"Expected action shape ({self.num_envs}, 8)")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Policy action contains NaN or Inf")
        if not (math.isfinite(max_step_m) and max_step_m > 0):
            raise ValueError("max_step_m must be positive and finite")

        # Limit changes in root-frame POSITION goals, preserving absolute IK format.
        delta = action[:, :3] - self.action[:, :3]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(max_step_m / distance.clamp_min(1e-8), max=1.0)
        action[:, :3] = self.action[:, :3] + delta * scale
        quat = action[:, 3:7]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if bool((norm < 1e-6).any()):
            raise ValueError("Policy quaternion norm is zero")
        action[:, 3:7] = quat / norm
        action[:, 7] = torch.where(action[:, 7] >= 0, 1.0, -1.0)

        # A positive gripper command opens the hand. Only the first opening
        # during an active attempt counts as that attempt's release event.
        opening = (mask & self.policy_attempt_active & ~self.policy_released
                   & (self.action[:, 7] < 0) & (action[:, 7] > 0))
        for i in torch.nonzero(opening).flatten().tolist():
            self.pickers[i].release()
        self._carrying &= ~opening
        self.policy_released |= opening
        self.policy_release_step = torch.where(
            opening, torch.full_like(self.policy_release_step, self.policy_step_index + 1),
            self.policy_release_step,
        )
        self.held_corner = torch.where(opening, -1, self.held_corner)
        self.held_vertex = torch.where(opening, -1, self.held_vertex)

        self.action = torch.where(mask[:, None], action, self.action)
        self.step()
        self.policy_step_index += 1
        self.target_w = self.tcp().clone()
        self.target_quat_w = as_tensor(self.ee.data.target_quat_w)[:, 0].clone()
        return {
            "released": opening.clone(),
            "lift_verified": self.policy_lift_verified.clone(),
            "dropped": self.dropped.clone(),
            "slipped": self.policy_slipped.clone(),
        }

    def _watch_transport(self):
        """Track lift, slip and drop using the ACTUAL selected grasp vertex."""
        point = self.held_grasp_point_position()
        active = self._carrying
        rise = point[:, 2] - self.policy_grip_start_z
        self.policy_lift_verified |= active & (rise >= self.check_min_rise)
        drift = torch.linalg.vector_norm((point - self.tcp()) - self.grasp_offset, dim=-1)
        self.max_drift = torch.where(active, torch.maximum(self.max_drift, drift), self.max_drift)
        self.policy_slipped |= active & (drift > self.slip_drift)
        dropped_now = cloth_dropped(
            point[:, 2], self.tcp()[:, 2], active,
            table_z=self.table_z, drop_height=self.drop_height,
            drop_hand_height=self.drop_hand_height,
        )
        self.dropped |= dropped_now
        self._carrying &= ~dropped_now

    def finish_policy_place(self, pairs, active=None, settle_seconds=1.0,
                            success_threshold=0.02):
        """Wait after policy release, then assess the current cloth fold.

        Call this AFTER active environments have issued their release action.
        This is a blocking, batch-level helper: suitable for integration
        tests. A real PPO environment needs nonblocking, per-env completion.

        'fold_success' means ALL target pairs are within threshold; after
        placing the first of two corners it will normally still be False.
        """
        mask = self._mask(active)
        if bool((mask & ~self.policy_released).any()):
            raise RuntimeError("Every active environment must release before finish_policy_place")
        if not (math.isfinite(settle_seconds) and settle_seconds >= 0):
            raise ValueError("settle_seconds must be nonnegative and finite")
        self.hold(math.ceil(settle_seconds / self.base.physics_dt))
        distances = pair_distances(self.corners(), pairs)
        success = fold_success(self.corners(), pairs, success_threshold)
        good = (mask & self.policy_attempt_active & self.policy_released
                & ~self.dropped & ~self.policy_slipped
                & (~self.physical | self.policy_lift_verified))
        return {
            "released": self.policy_released & mask,
            "valid_attempt": good,
            "fold_success": success & good,
            "pair_distances_m": distances,
            "dropped": self.dropped & mask,
            "slipped": self.policy_slipped & mask,
            "lift_verified": self.policy_lift_verified & mask,
            "max_drift_m": self.max_drift.clone(),
        }

    def hold(self, count):
        for _ in range(count):
            self.step()

    def move(self, goal_w, count=240, active=None, goal_quat_w=None, max_wait=1200):
        """Move the hands of active environments to goal_w (N, 3); others keep their target.

        Returns the remaining TCP error per environment (N,) [m]; inactive environments report 0.
        Unlike the demo this does not raise on a miss, so one environment cannot stop the others.
        max_wait: extra steps to let the hand arrive (the demo uses 60 when the fingers may touch something).
        """
        active = self._mask(active)
        goal = torch.where(active[:, None], goal_w, self.target_w)
        goal_quat = self.target_quat_w.clone()
        if goal_quat_w is not None:
            goal_quat = torch.where(active[:, None], goal_quat_w.expand_as(goal_quat), goal_quat)
        flip = (self.target_quat_w * goal_quat).sum(-1, keepdim=True) < 0  # q and -q are the same rotation
        goal_quat = torch.where(flip, -goal_quat, goal_quat)

        positions = smoothstep_path(self.target_w, goal, count)
        quats = smoothstep_path(self.target_quat_w, goal_quat, count)
        for position, quat in zip(positions, quats):
            self._write_pose(position, quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True))
            self.step()
        self.target_w = goal.clone()
        self.target_quat_w = goal_quat / torch.linalg.vector_norm(goal_quat, dim=-1, keepdim=True)
        for _ in range(max_wait):  # fixed-duration interpolation alone does not guarantee arrival
            error = torch.linalg.vector_norm(self.tcp() - goal, dim=-1) * active
            if bool((error < REACH_TOLERANCE).all()):
                break
            self.step()
        return torch.linalg.vector_norm(self.tcp() - goal, dim=-1) * active

    def set_gripper(self, closed, active=None):
        """Open/close in active environments. Returns the finger gap per environment after it (N,) [m]."""
        active = self._mask(active)
        # BinaryJointPositionAction: negative = close, positive = open
        command = torch.full_like(self.action[:, 7], -1.0 if closed else 1.0)
        self.action[:, 7] = torch.where(active, command, self.action[:, 7])
        self.hold(max(60, math.ceil(1.0 / self.base.physics_dt)))
        finger_ids = self.robot.find_joints("panda_finger_joint.*")[0]
        return as_tensor(self.robot.data.joint_pos)[:, finger_ids].sum(dim=-1)

    # ---------- one decision ----------
    def grasp_corner(self, source, active=None, pick_xy=None, policy_controls_lift=False):
        """Pick corner number source (N,) in active environments.

        Returns ok (N,) bool and info {"attempts": (N,), "rise_m": (N,), "corner_tcp_m": (N,)}.
        Physical: grasp check after closing, one retry by default; environments that already
        hold their corner wait while the others retry.

        If pick_xy is supplied, it is the policy-selected WORLD XY location.
        The script approaches its closest cloth vertex while source remains
        the logical corner index used for fold evaluation.

        If policy_controls_lift=True, the script stops after closing the fingers
        and settling. The policy handles lift/carry/release. In this mode ok
        means the grasp attempt was initialized, NOT physically verified.
        The caller must check lift/slip/drop during policy rollout.
        """
        active = self._mask(active)
        source = torch.as_tensor(source, device=self.device, dtype=torch.long)
        if source.shape != (self.num_envs,) or bool(((source < 0) | (source > 3)).any()):
            raise ValueError(f"source must be a ({self.num_envs},) tensor with corner indices 0..3")
        if pick_xy is not None:
            policy_target, policy_vertex_ids = self.policy_grasp_target(pick_xy)
        else:
            policy_target, policy_vertex_ids = None, None
        n = self.num_envs
        ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        attempts = torch.zeros(n, dtype=torch.long, device=self.device)
        rise = torch.zeros(n, device=self.device)
        corner_tcp = torch.zeros(n, device=self.device)
        for _ in range(1 + self.retries):
            trying = active & ~ok
            if not bool(trying.any()):
                break
            attempts += trying.long()
            above = self.grasp_target(source) if policy_target is None else policy_target.clone()
            above[:, 2] += APPROACH_HEIGHT
            safe = self.tcp().clone()
            safe[:, 2] = torch.maximum(safe[:, 2], above[:, 2])  # rise vertically before crossing the cloth
            self.move(safe, 120, trying)
            self.move(above, 240, trying, goal_quat_w=self.action.new_tensor(DOWN_QUAT))
            touch = self.grasp_target(source) if policy_target is None else policy_target.clone()
            # Physical: fingertips on the cloth (the table stops them). Virtual: 1.5 cm above the corner.
            touch[:, 2] += -self.press_depth if self.physical else TOUCH_HEIGHT
            reach_error = self.move(touch, 180, trying, max_wait=60 if self.physical else 1200)
            self.set_gripper(True, trying)

            attached = torch.zeros(n, dtype=torch.bool, device=self.device)
            corner = self.corners()[torch.arange(n, device=self.device), source]
            for i in torch.nonzero(trying).flatten().tolist():
                if not self.physical and reach_error[i] >= REACH_TOLERANCE:
                    continue
                vertex = int(self.corner_indices[i, source[i]]) if policy_vertex_ids is None else int(policy_vertex_ids[i])
                position = self.nodes()[i, vertex]
                if self.pickers[i].grasp(position, vertex_id=vertex):
                    attached[i] = True
                    self.held_corner[i] = source[i]
                    self.held_vertex[i] = vertex
            if self.physical and not policy_controls_lift:
                passed, rise_now, corner_tcp_now = self._grasp_check(attached)
                rise = torch.where(attached, rise_now, rise)
                corner_tcp = torch.where(attached, corner_tcp_now, corner_tcp)
            else:
                # No automatic lift: proximity/vertex registration is not proof
                # of a successful physical grasp. The policy must lift to verify.
                if self.physical:
                    self.hold(self.grasp_settle_steps)
                passed = attached
            ok |= passed
            # Remember the corner-to-hand offset from the settled grip
            # Track actual selected grip point for slippage, but keep the old
            # logical corner offset for the legacy scripted placement path.
            self.offset = torch.where(passed[:, None], self.held_corner_position() - self.tcp(), self.offset)
            self.grasp_offset = torch.where(passed[:, None], self.held_grasp_point_position() - self.tcp(), self.grasp_offset)
            self._carrying |= passed
            # A newly initialized grasp starts a new policy-controlled carry.
            self.policy_attempt_active |= passed
            self.policy_released &= ~passed
            self.policy_release_step = torch.where(passed, -1, self.policy_release_step)
            self.policy_lift_verified &= ~passed
            self.policy_slipped &= ~passed
            self.dropped &= ~passed
            self.max_drift = torch.where(passed, 0.0, self.max_drift)
            self.policy_grip_start_z = torch.where(
                passed, self.held_grasp_point_position()[:, 2], self.policy_grip_start_z)
            failed = trying & ~passed
            if bool(failed.any()):  # let go and back off before the retry
                for i in torch.nonzero(failed).flatten().tolist():
                    self.pickers[i].release()
                self.held_corner = torch.where(failed, torch.full_like(self.held_corner, -1), self.held_corner)
                self.held_vertex = torch.where(failed, -1, self.held_vertex)
                self.set_gripper(False, failed)
                retreat = self.tcp().clone()
                retreat[:, 2] += 0.05
                self.move(retreat, 120, failed, max_wait=60)
        self.check_rise = torch.where(ok, rise, self.check_rise)
        verified = ok & (not self.physical or not policy_controls_lift)
        return ok, {"attempts": attempts, "rise_m": rise, "corner_tcp_m": corner_tcp,
                    "grasp_verified": verified}

    def _grasp_check(self, active):
        """Physical: settle, lift check_lift slowly, hold; the corner must follow. Same rule as the demo."""
        self.hold(self.grasp_settle_steps)
        start_z = self.held_grasp_point_position()[:, 2].clone()
        lift = self.tcp().clone()
        lift[:, 2] += self.check_lift
        self.move(lift, self.check_lift_steps, active, max_wait=60)
        self.hold(self.check_hold_steps)
        corner = self.held_grasp_point_position()
        rise = corner[:, 2] - start_z
        corner_tcp = torch.linalg.vector_norm(corner - self.tcp(), dim=-1)
        passed = active & (rise >= self.check_min_rise)
        if self.quality_max > 0:
            passed &= corner_tcp <= self.quality_max
        return passed, rise, corner_tcp

    def place_held_corner(self, place_xy, active=None):
        """Carry the held corner so it lands at place_xy (N, 2), release, retreat, settle.

        Only environments that are active AND holding a corner move. Returns the signed xy miss of the
        corner after settling (N, 2) [m] (0 where nothing was placed) and info
        {"placed": (N,) bool, "dropped": (N,) bool, "max_drift_m": (N,)}.
        """
        moving = self._mask(active) & (self.held_corner >= 0)
        rows = torch.arange(self.num_envs, device=self.device)
        source = self.held_corner.clamp(min=0)
        place_point = place_point_from_xy(place_xy.to(self.device), self.table_z, self.cloth_thickness)
        start = self.held_corner_position().clone()
        if self.physical:
            # The grasp check already lifted the corner: start the arc from the height it had before (demo)
            start[:, 2] -= self.check_rise
        self.dropped = torch.where(moving, torch.zeros_like(self.dropped), self.dropped)
        self.max_drift = torch.where(moving, torch.zeros_like(self.max_drift), self.max_drift)
        self._carrying = moving.clone()

        for point in carry_arc_corner_points(start, place_point, self.lift_height, self.cloth_thickness):
            if self.physical:  # the corner slides in the fingers, so use where it is now
                self.offset = torch.where(self._carrying[:, None], self.held_corner_position() - self.tcp(),
                                          self.offset)
            self.move(point - self.offset, 120, self._carrying)
        # Physical: lower three times with a fresh offset (the fingers rest on the cloth below, the hand
        # cannot reach the goal). Virtual: the offset is fixed, so one lowering is enough.
        for _ in range(3 if self.physical else 1):
            if self.physical:
                self.offset = torch.where(self._carrying[:, None], self.held_corner_position() - self.tcp(),
                                          self.offset)
            self.move(lower_goal(place_point, self.offset, self.cloth_thickness), 120, self._carrying,
                      max_wait=60 if self.physical else 1200)
        placed = self._carrying.clone()
        self._carrying.zero_()
        self.set_gripper(False, moving)
        for i in torch.nonzero(moving).flatten().tolist():
            self.pickers[i].release()
        self.held_corner = torch.where(moving, torch.full_like(self.held_corner, -1), self.held_corner)
        self.held_vertex = torch.where(moving, -1, self.held_vertex)
        retreat = self.tcp().clone()
        retreat[:, 2] += RETREAT_HEIGHT
        self.move(retreat, 180, moving)
        self.hold(self.settle_steps)

        miss = (self.corners()[rows, source, :2] - place_point[:, :2]) * placed[:, None]
        return miss, {"placed": placed, "dropped": self.dropped.clone(), "max_drift_m": self.max_drift.clone()}

    def release_all(self):
        """Drop every grasp; call before env.reset()."""
        for picker in self.pickers:
            picker.release()
        self.held_corner.fill_(-1)
        self.held_vertex.fill_(-1)
        self._carrying.zero_()
        self.policy_attempt_active.zero_()
        self.policy_released.zero_()
        self.policy_lift_verified.zero_()
        self.policy_slipped.zero_()
        self.policy_release_step.fill_(-1)
        self.policy_grip_start_z.zero_()
        self.policy_step_index = 0

    def _mask(self, active):
        if active is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return active.to(self.device).bool()
