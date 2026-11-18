"""Collect manual pick/place episodes using the PR #27 dataset format.

Call append_step after env.step, using the action actually sent to the robot.
Inputs describe ONE environment, with no leading environment dimension.
"""

import numpy as np
import torch

from scripts.learning.demonstration_recorder import DemonstrationRecorder
from scripts.learning.reward import check_fold_success


def _array(value, shape):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array of shape {shape}, got {result.shape}")
    return result


class TeleopRecorder:
    """One episode, absolute pose IK actions, and explicit pick/place boundaries.

    action: robot-root-frame [x, y, z, qw, qx, qy, qz, gripper].
    corners_w and tcp_w: world coordinates in metres.
    Segment indices use Python's half-open [start, end) convention.
    """

    def __init__(self, demo_dir, pairs, success_threshold=0.02):
        self.recorder = DemonstrationRecorder(demo_dir, pairs, success_threshold)
        self.rows = []
        self._pending = None
        self._finished = False

    def _check_open(self):
        if self._finished:
            raise RuntimeError("Episode finished; create a new recorder after reset")

    def append_step(self, time_s, phase, action, joint_pos, corners_w, tcp_w, grasping):
        self._check_open()
        time_s = float(time_s)
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("time_s must be finite and nonnegative")
        if self.rows and time_s <= self.rows[-1][0]:
            raise ValueError("time_s must increase each recorded step")
        self.rows.append((
            time_s, str(phase), _array(action, (8,)), _array(joint_pos, (9,)),
            _array(corners_w, (4, 3)), _array(tcp_w, (3,)), bool(grasping),
        ))

    def begin_pick_place(self, corners_w, pick_xy):
        """Call before the first recorded step of a manual pick/place attempt."""
        self._check_open()
        if self._pending is not None:
            raise RuntimeError("Previous pick/place has not ended")
        self._pending = (
            torch.from_numpy(_array(corners_w, (4, 3))),
            _array(pick_xy, (2,)), len(self.rows),
        )

    def end_pick_place(self, corners_w, place_xy):
        """Call after release/settling; place_xy is the intended place target."""
        self._check_open()
        if self._pending is None:
            raise RuntimeError("No active pick/place")
        before, pick, start = self._pending
        if start == len(self.rows):
            raise ValueError("Pick/place must contain at least one low-level sample")
        result = self.recorder.record_action(
            before_corners=before,
            after_corners=torch.from_numpy(_array(corners_w, (4, 3))),
            pick_xy=pick.tolist(), place_xy=_array(place_xy, (2,)).tolist(),
            low_level_start=start, low_level_end=len(self.rows),
        )
        self._pending = None
        return result

    def finish(self, final_corners_w, episode_id, *, aborted=False):
        """Save only successful, completed episodes; return whether saved.

        final_corners_w must be measured after release, retreat, and settling.
        Aborted episodes are discarded, including an unfinished pick/place.
        """
        self._check_open()
        if aborted:
            self._finished = True
            return False
        if self._pending is not None:
            raise RuntimeError("End the active pick/place before finishing")
        final = torch.from_numpy(_array(final_corners_w, (4, 3)))
        success = check_fold_success(
            final, self.recorder.pairs, self.recorder.success_threshold,
        )
        self.recorder.finalize(final, success)
        saved = self.recorder.save_if_success(success, self.rows, episode_id)
        self._finished = True
        return saved
