"""Stage 6: batched hybrid pick/low-level rollout bridge for BatchedFoldExecutor.

This is NOT a full PPO trainer or Gym environment. Its API splits a decision into:
  begin_pick(pick_xy): pick policy chooses world XY, scripted IK closes fingers;
  step_low_level(action): low-level actor supplies absolute root-frame 8D IK goals.

For a first integration test, source corner order is fixed by `pairs` (robot-near
first, then second). Done environments remain masked until a *full* scene reset.
The executor's scripted grasp is blocking for all parallel environments; therefore
this bridge is not yet suitable for high-throughput asynchronous PPO rollouts.
"""
from __future__ import annotations

import math
import torch

from scripts.rl.fold_mdp import fold_reward, fold_success, pair_distances


class HybridPPOBridge:
    PICK = 0
    CARRY = 1
    SETTLE = 2
    DONE = 3

    def __init__(self, executor, pairs, *, success_threshold=0.02,
                 settle_seconds=1.0, max_low_steps=1000,
                 shape_success_fn=None):
        if len(pairs) != 2:
            raise ValueError("Expected exactly two ordered source-destination pairs")
        if success_threshold <= 0 or settle_seconds < 0 or max_low_steps < 1:
            raise ValueError("Invalid threshold, settle duration, or step limit")
        self.ex = executor
        self.pairs = [(int(a), int(b)) for a, b in pairs]
        self.n, self.device = executor.num_envs, executor.device
        self.threshold = float(success_threshold)
        self.settle_steps = math.ceil(settle_seconds / executor.base.physics_dt)
        self.max_low_steps = max_low_steps
        self.shape_success_fn = shape_success_fn
        self.reset_tracking()

    def reset_tracking(self):
        """Only call AFTER resetting Isaac and constructing/resetting the executor.

        Executor reset semantics are not implemented here; this resets ONLY bridge counters.
        """
        self.phase = torch.full((self.n,), self.PICK, device=self.device, dtype=torch.long)
        self.pair_index = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.low_steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.wait_steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.invalid = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.prev_corners = self.ex.corners().clone()

    def observation(self):
        """Tensor observations for both policies; all coordinates are WORLD except ik_action.

        Do not change order or coordinate frame without changing BC preprocessing.
        """
        n = self.n
        return {
            "corners_w": self.ex.corners().clone(),           # (N,4,3)
            "tcp_w": self.ex.tcp().clone(),                   # (N,3)
            "ik_action_root": self.ex.action.clone(),        # (N,8)
            "phase": self.phase.clone(),                     # (N,)
            "pair_index": self.pair_index.clone(),           # (N,)
            "grasping": (self.ex.held_vertex >= 0).clone(),   # (N,)
            "pair_distances_m": pair_distances(self.ex.corners(), self.pairs),
        }

    def begin_pick(self, pick_xy):
        """Execute scripted approach/close for envs requesting pick (BLOCKING).

        pick_xy is (N,2) WORLD coordinates. For non-PICK envs input is ignored.
        A physical return of ok=True is only an initialized grasp attempt;
        successful physical hold is verified during learned lift.
        """
        pick_xy = torch.as_tensor(pick_xy, device=self.device, dtype=self.ex.corners().dtype)
        if pick_xy.shape != (self.n, 2) or not bool(torch.isfinite(pick_xy).all()):
            raise ValueError(f"pick_xy must be finite with shape ({self.n}, 2)")
        active = self.phase == self.PICK
        if not bool(active.any()):
            raise RuntimeError("No environment awaiting a pick")
        # Source order is fixed; policy chooses WHERE inside that corner region to grasp.
        sources = torch.tensor([p[0] for p in self.pairs], device=self.device)
        source = sources[self.pair_index.clamp(max=1)]
        # Note: executor.policy_grasp_target currently validates ALL rows, even inactive.
        # Supply the legacy target in inactive rows so only active selection matters.
        xy = pick_xy.clone()
        if bool((~active).any()):
            fallback = self.ex.grasp_target(source)[:, :2]
            xy[~active] = fallback[~active]
        ok, info = self.ex.grasp_corner(source, active=active, pick_xy=xy, policy_controls_lift=True)
        self.invalid |= active & ~ok
        self.phase = torch.where(active & ok, self.CARRY, self.phase)
        self.phase = torch.where(active & ~ok, self.DONE, self.phase)
        self.low_steps = torch.where(active, 0, self.low_steps)
        # The scripted grasp takes physics steps and can move the cloth; avoid rewarding
        # those steps as if they were low-level policy actions.
        self.prev_corners = self.ex.corners().clone()
        return self.observation(), {"pick_initialized": ok, "details": info}

    def step_low_level(self, action):
        """Advance ONE physics step for all N envs, return (obs, reward, done, info).

        Active policy actions are expected to contain a correctly framed IK quaternion.
        Reward is assigned ONLY for low-level steps; for now release settling is
        nonblocking, and no actor action is applied to settling/done envs.
        """
        if not bool(((self.phase == self.CARRY) | (self.phase == self.SETTLE)).any()):
            raise RuntimeError("No carrying or settling environment; call begin_pick first")
        active = self.phase == self.CARRY
        before = self.ex.corners().clone()
        # During settling no policy action is needed. Supply the last valid IK
        # command; inactive rows must not force the actor to emit valid quats.
        command = self.ex.action.clone()
        if bool(active.any()):
            incoming = torch.as_tensor(action, device=self.device, dtype=command.dtype)
            if incoming.shape != command.shape:
                raise ValueError(f"Expected action shape {tuple(command.shape)}")
            command[active] = incoming[active]
        status = self.ex.step_policy_action(command, active=active)
        after = self.ex.corners().clone()
        self.low_steps += active.long()
        released = active & status["released"]
        failure = active & (status["dropped"] | status["slipped"])
        timeout = active & (self.low_steps >= self.max_low_steps)
        # Released environments wait for cloth to settle before proceeding.
        self.phase = torch.where(released, self.SETTLE, self.phase)
        self.wait_steps = torch.where(released, 0, self.wait_steps)
        self.phase = torch.where(failure | timeout, self.DONE, self.phase)
        self.invalid |= failure | timeout

        settling = self.phase == self.SETTLE
        self.wait_steps += settling.long()
        completed = settling & (self.wait_steps >= max(1, self.settle_steps))
        self.invalid |= completed & self.ex.physical & ~status["lift_verified"]
        # First successfully completed placement -> next pick; second -> episode done.
        good = completed & ~self.invalid
        next_pair = good & (self.pair_index == 0)
        final_pair = good & (self.pair_index == 1)
        self.pair_index += next_pair.long()
        self.phase = torch.where(next_pair, self.PICK, self.phase)
        self.phase = torch.where(final_pair | (completed & self.invalid), self.DONE, self.phase)

        all_pairs_success = fold_success(after, self.pairs, self.threshold)
        success = final_pair & all_pairs_success
        if self.shape_success_fn is not None:
            for i in torch.where(success)[0].tolist():
                success[i] = bool(self.shape_success_fn(i))
        # Final success bonus only once, on completion; no false repeated bonus.
        reward = fold_reward(before, after, self.pairs, success, failure)
        reward = torch.where(active | completed, reward, torch.zeros_like(reward))
        done = self.phase == self.DONE
        self.prev_corners = after
        return self.observation(), reward, done, {
            "phase": self.phase.clone(), "released": released,
            "pick_required": self.phase == self.PICK,
            "fold_success": success, "invalid": self.invalid.clone(),
            "timeout": timeout, "pair_index": self.pair_index.clone(),
            "status": status,
        }
