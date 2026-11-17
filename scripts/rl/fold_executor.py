"""Run "grasp a corner" and "place the held corner at (x, y)" in N environments at once.

Needs Isaac Sim (run through the Isaac Lab launcher). Virtual grasp only for now.

Why every environment moves in lockstep: Isaac Lab steps all N environments in one physics
scene, and env.step() takes one action row per environment. So each call below runs the same
number of physics steps in every environment. Environments that are not taking part
(active = False) keep their hand where it is and just wait.

Typical use (see run_executor_smoke.py):
    configure_virtual_fold_cfg(cfg, num_envs)          # before gym.make
    executor = BatchedFoldExecutor(env, ...)            # after env.reset()
    ok = executor.grasp_corner(source)                  # source: (N,) corner number 0-3
    executor.place_held_corner(place_xy, active=ok)     # place_xy: (N, 2) where the corner should land
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

from a_single_point_picker import as_tensor  # noqa: E402
from cloth_corner_tracker import detect_corner_vertex_indices  # noqa: E402
from virtual_attachment_picker import create_picker  # noqa: E402

from scripts.rl.fold_motion import (  # noqa: E402
    APPROACH_HEIGHT, RETREAT_HEIGHT, TOUCH_HEIGHT,
    carry_arc_goals, lower_goal, place_point_from_xy, smoothstep_path,
)

REACH_TOLERANCE = 0.005  # m, same as the demo's move()
DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)  # world hand orientation the demo uses for the approach


def filter_gripper_contacts_all_envs(env, env_ids=None):
    """Virtual grasp: stop hand/finger shapes from colliding with the cloth, in EVERY environment.

    Same as virtual_attachment_picker.filter_gripper_contacts, which only handles environment 0.
    Must run before physics starts (EventTermCfg mode="prestartup").
    """
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage
    hand_names = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    count = 0
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
        count += 1
    print(f"[executor] filtered gripper-cloth contacts in {count} environments", flush=True)


def configure_virtual_fold_cfg(cfg, num_envs):
    """Task settings the virtual fold needs, copied from rectangular_fold_demo.py (virtual grasp).

    Cloth material and size are left to the caller.
    """
    from isaaclab.managers import EventTermCfg

    cfg.scene.num_envs = num_envs
    cfg.decimation = 1             # one env.step = one physics step, so the picker updates every step
    cfg.episode_length_s = 300.0   # long enough that time_out never cuts a fold short
    cfg.scene.support_neg_y = None
    cfg.scene.support_pos_y = None
    cfg.scene.table.spawn.visible = True
    cfg.events.reset_deformable = None
    cfg.events.virtual_grasp_contacts = EventTermCfg(func=filter_gripper_contacts_all_envs, mode="prestartup")
    cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
    cfg.curriculum.gravity = None
    cfg.curriculum.action_rate = None
    cfg.commands.deformable_pose.debug_vis = False
    cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    cfg.actions = type(cfg)().actions.ik
    cfg.actions.arm_action.body_offset.pos = (0.0, 0.0, 0.1034)  # measured TCP used by ee_frame
    cfg.actions.arm_action.controller.ik_params = {"lambda_val": 0.1}
    for name in vars(cfg.terminations):
        if name != "time_out":
            setattr(cfg.terminations, name, None)
    return cfg


class BatchedFoldExecutor:
    def __init__(self, env, *, table_z=0.0, cloth_thickness=0.003, lift_height=0.12,
                 grasp_radius=0.03, settle_seconds=3.0):
        self.env = env
        self.base = env.unwrapped
        scene = self.base.scene
        self.cloth, self.ee, self.robot = scene["deformable"], scene["ee_frame"], scene["robot"]
        self.num_envs = self.base.num_envs
        self.device = self.base.device
        self.table_z = table_z
        self.cloth_thickness = cloth_thickness
        self.lift_height = lift_height
        self.settle_steps = math.ceil(settle_seconds / self.base.physics_dt)

        self.action = torch.zeros(self.base.action_space.shape, device=self.device)
        if self.action.shape != (self.num_envs, 8):
            raise RuntimeError(f"Expected absolute pose IK action (N, 8), got {tuple(self.action.shape)}")
        self.target_w = self.tcp().clone()
        self.target_quat_w = as_tensor(self.ee.data.target_quat_w)[:, 0].clone()
        self.action[:, 7] = 1.0  # gripper open
        self._write_pose()

        # One picker per environment; each attaches one vertex of its own cloth
        self.pickers = [create_picker(self.cloth, (lambda i=i: self.tcp()[i]), env_id=i, grasp_radius=grasp_radius)
                        for i in range(self.num_envs)]
        self.offset = torch.zeros(self.num_envs, 3, device=self.device)  # held corner - TCP
        self.held_corner = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.corner_indices = None  # (N, 4) mesh vertex of each corner, found on the flat cloth

    # ---------- reading state ----------
    def nodes(self):
        return as_tensor(self.cloth.data.nodal_pos_w)  # (N, V, 3)

    def tcp(self):
        return as_tensor(self.ee.data.target_pos_w)[:, 0]  # (N, 3)

    def detect_corners(self):
        """Call once on the flat cloth (after env.reset() and a short hold)."""
        nodes = self.nodes()
        self.corner_indices = torch.stack([detect_corner_vertex_indices(nodes[i]) for i in range(self.num_envs)])
        return self.corner_indices

    def corners(self):
        """(N, 4, 3) world positions of the four tracked corners."""
        nodes = self.nodes()
        return torch.gather(nodes, 1, self.corner_indices.unsqueeze(-1).expand(-1, -1, 3))

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
            picker.update(dt=self.base.physics_dt)  # attachment backend: update BEFORE the physics step
        _, _, terminated, truncated, _ = self.env.step(self.action)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("An environment ended during a pick-and-place; increase episode_length_s")

    def hold(self, count):
        for _ in range(count):
            self.step()

    def move(self, goal_w, count=240, active=None, goal_quat_w=None, max_wait=1200):
        """Move the hands of active environments to goal_w (N, 3); others keep their target.

        Returns the remaining TCP error per environment (N,) [m]; inactive environments report 0.
        Unlike the demo this does not raise on a miss, so one environment cannot stop the others.
        """
        active = self._mask(active)
        goal = torch.where(active[:, None], goal_w, self.target_w)
        goal_quat = self.target_quat_w.clone()
        if goal_quat_w is not None:
            goal_quat = torch.where(active[:, None], goal_quat_w.expand_as(goal_quat), goal_quat)
        # Take the short way round (q and -q are the same rotation)
        flip = (self.target_quat_w * goal_quat).sum(-1, keepdim=True) < 0
        goal_quat = torch.where(flip, -goal_quat, goal_quat)

        positions = smoothstep_path(self.target_w, goal, count)
        quats = smoothstep_path(self.target_quat_w, goal_quat, count)
        for position, quat in zip(positions, quats):
            self._write_pose(position, quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True))
            self.step()
        self.target_w, self.target_quat_w = goal.clone(), goal_quat / torch.linalg.vector_norm(
            goal_quat, dim=-1, keepdim=True)
        # Fixed-duration interpolation alone does not guarantee arrival
        for _ in range(max_wait):
            error = torch.linalg.vector_norm(self.tcp() - goal, dim=-1) * active
            if bool((error < REACH_TOLERANCE).all()):
                break
            self.step()
        return torch.linalg.vector_norm(self.tcp() - goal, dim=-1) * active

    def set_gripper(self, closed, active=None):
        active = self._mask(active)
        # BinaryJointPositionAction: negative = close, positive = open
        self.action[:, 7] = torch.where(active, torch.full_like(self.action[:, 7], -1.0 if closed else 1.0),
                                        self.action[:, 7])
        self.hold(max(60, math.ceil(1.0 / self.base.physics_dt)))

    # ---------- one decision ----------
    def grasp_corner(self, source, active=None):
        """Pick corner number source (N,) in active environments. Returns ok (N,) bool.

        Same steps as the demo: lift to a safe height, approach 8 cm above, descend to 1.5 cm above,
        close, attach. An environment whose corner is out of reach returns False (the demo raises).
        """
        active = self._mask(active)
        source = source.to(self.device).long()
        rows = torch.arange(self.num_envs, device=self.device)
        corner = self.corners()[rows, source]
        above = corner.clone()
        above[:, 2] += APPROACH_HEIGHT
        safe = self.tcp().clone()
        safe[:, 2] = torch.maximum(safe[:, 2], above[:, 2])  # rise vertically before crossing the cloth
        self.move(safe, 120, active)
        self.move(above, 240, active, goal_quat_w=self.action.new_tensor(DOWN_QUAT))
        touch = self.corners()[rows, source].clone()
        touch[:, 2] += TOUCH_HEIGHT
        reach_error = self.move(touch, 180, active)
        self.set_gripper(True, active)

        ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        corner = self.corners()[rows, source]
        for i in torch.nonzero(active).flatten().tolist():
            vertex = int(self.corner_indices[i, source[i]])
            if reach_error[i] < REACH_TOLERANCE and self.pickers[i].grasp(corner[i], vertex_id=vertex):
                ok[i] = True
                self.offset[i] = self.pickers[i].target_position_w - self.tcp()[i]
                self.held_corner[i] = source[i]
        return ok

    def place_held_corner(self, place_xy, active=None):
        """Carry the held corner so it lands at place_xy (N, 2), release, retreat, settle.

        Only environments that are active AND holding a corner move. Returns the signed xy miss of
        the held corner after settling (N, 2) [m] (0 for environments that did not place).
        """
        active = self._mask(active) & (self.held_corner >= 0)
        rows = torch.arange(self.num_envs, device=self.device)
        source = self.held_corner.clamp(min=0)
        place_point = place_point_from_xy(place_xy.to(self.device), self.table_z, self.cloth_thickness)
        start = self.corners()[rows, source].clone()

        for goal in carry_arc_goals(start, place_point, self.offset, self.lift_height, self.cloth_thickness):
            self.move(goal, 120, active)
        for _ in range(3):  # the demo lowers three times; the cloth under the corner keeps settling
            self.move(lower_goal(place_point, self.offset, self.cloth_thickness), 120, active)
        self.set_gripper(False, active)
        for i in torch.nonzero(active).flatten().tolist():
            self.pickers[i].release()
        self.held_corner = torch.where(active, torch.full_like(self.held_corner, -1), self.held_corner)
        retreat = self.tcp().clone()
        retreat[:, 2] += RETREAT_HEIGHT
        self.move(retreat, 180, active)
        self.hold(self.settle_steps)

        miss = self.corners()[rows, source, :2] - place_point[:, :2]
        return miss * active[:, None]

    def release_all(self):
        """Drop every attachment; call before env.reset()."""
        for picker in self.pickers:
            picker.release()
        self.held_corner.fill_(-1)

    def _mask(self, active):
        if active is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return active.to(self.device).bool()
