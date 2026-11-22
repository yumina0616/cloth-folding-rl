"""Save a scripted fold run in the BC dataset format read by PR #37.

Files written into the run folder (successful runs only), the same four files as the keyboard teleop:
  high_level.jsonl  one record per pick/place (corners before, pick/place XY, low-level segment)
  low_level.npz     every physics step: time_s, phase, action, joint_pos, corners_w, tcp_w, grasping
  metadata.json     written by the shared DemonstrationRecorder
  context.json      conventions (frames, units, pick/place meaning), as in teleop_cloth.py

Phase labels in low_level.npz follow the loader (scripts/learning/demonstration_dataset.py):
  ready             scripted motion without cloth (approach, descend, retreat, settle)
  close             fingers closing; the PR #37 executor scripts this part
  carry             fingers closed on the cloth: grasp-check lift, carry arc, lowering
  physical_release  fingers opening at the place point
  grasp_retry       a grasp attempt whose grasp check failed (kept, but not trained as carry)
The script's own detailed phase names are saved next to it in low_level_phase_detail.npy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _np(value, shape):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {result.shape}")
    return result


class FoldEpisodeLog:
    """Collects one run; nothing is written until save()."""

    def __init__(self):
        self.rows = []      # [time_s, label, action(8), joint_pos(9), corners(4,3), tcp(3), grasping]
        self.details = []   # the script's phase name per row
        self.segments = []  # completed pick/place attempts
        self.pending = None
        self.label = "ready"

    def append(self, time_s, phase_detail, action, joint_pos, corners_w, tcp_w, grasping):
        self.rows.append([float(time_s), self.label, _np(action, (8,)), _np(joint_pos, (9,)),
                          _np(corners_w, (4, 3)), _np(tcp_w, (3,)), bool(grasping)])
        self.details.append(str(phase_detail))

    def begin(self, corners_w, pick_xy):
        """Right before the close command of a grasp attempt."""
        if self.pending is not None:
            raise RuntimeError("Previous grasp attempt was neither ended nor abandoned")
        self.pending = {"start": len(self.rows), "before": _np(corners_w, (4, 3)), "pick": _np(pick_xy, (2,))}
        self.label = "close"

    def start_carry(self):
        """The fingers are closed on the cloth; from here the low-level policy is in charge."""
        self.label = "carry"

    def abandon(self):
        """The grasp check failed: relabel this attempt so it is not trained as carry."""
        if self.pending is not None:
            for row in self.rows[self.pending["start"]:]:
                row[1] = "grasp_retry"
        self.pending = None
        self.label = "ready"

    def begin_release(self, place_xy):
        """Right before the open command at the place point."""
        if self.pending is None:
            raise RuntimeError("No grasp attempt to release")
        self.pending["place"] = _np(place_xy, (2,))
        self.label = "physical_release"

    def end(self, corners_w, pair_index, correction=False):
        """After the fingers have opened."""
        if self.pending is None or "place" not in self.pending:
            raise RuntimeError("end() needs begin() and begin_release() first")
        if len(self.rows) <= self.pending["start"]:
            raise ValueError("A pick/place must contain at least one recorded step")
        self.segments.append({**self.pending, "end": len(self.rows), "after": _np(corners_w, (4, 3)),
                              "pair_index": int(pair_index), "correction": bool(correction)})
        self.pending = None
        self.label = "ready"

    def save(self, run_dir, pairs, threshold, final_corners_w, episode_id, context):
        """Replay the run into the shared TeleopRecorder; returns True when the files were written."""
        import torch
        from scripts.learning.teleop_recorder import TeleopRecorder

        if self.pending is not None:
            raise RuntimeError("A grasp attempt is still open")
        run_dir = Path(run_dir)
        recorder = TeleopRecorder(run_dir, [tuple(p) for p in pairs], threshold)
        starts = {s["start"]: s for s in self.segments}
        ends = {s["end"]: s for s in self.segments}
        for index, row in enumerate(self.rows):
            if index in starts:
                recorder.begin_pick_place(starts[index]["before"], starts[index]["pick"])
            recorder.append_step(*row)
            if index + 1 in ends:
                segment = ends[index + 1]
                recorder.end_pick_place(segment["after"], segment["place"])
                # Extra keys for the loader: which pair this pick served, and whether it was a correction
                record = recorder.recorder.logger.records[-1]
                record["pair_index"] = segment["pair_index"]
                record["correction"] = segment["correction"]
        saved = recorder.finish(torch.as_tensor(np.asarray(final_corners_w, dtype=float)), episode_id)
        if saved:
            np.save(run_dir / "low_level_phase_detail.npy", np.array(self.details))
            (run_dir / "context.json").write_text(json.dumps(context, default=str, indent=2), encoding="utf-8")
        return saved


def fold_context(*, grasp_mode, corner_vertex_indices, pairs, sample_dt, success_threshold_m,
                 final_pair_distances_m, config):
    """context.json with the same keys and conventions as teleop_cloth.py."""
    return {
        "source": "scripted_rectangular_fold_demo", "grasp_mode": grasp_mode,
        "action_type": "absolute_pose_ik", "action_frame": "robot_root",
        "quaternion_order": "wxyz", "position_unit": "metre", "time_unit": "second",
        "gripper_open": 1, "gripper_close": -1,
        "corner_vertex_indices": list(corner_vertex_indices), "pairs": [list(p) for p in pairs],
        "sample_dt": sample_dt, "success_threshold_m": success_threshold_m,
        "final_pair_distances_m": final_pair_distances_m,
        # The loader checks these exact strings. In this script the close is scripted, not manual.
        "pick_xy": ("measured TCP at manual close" if grasp_mode == "physical"
                    else "measured cloth corner at grasp"),
        "grasping_semantics": ("picker attached after the scripted close; the grasp check decides the attempt"
                               if grasp_mode == "physical" else "virtual attachment active"),
        "place_xy": "measured TCP immediately before opening",
        "phase_labels": "ready / close / carry / physical_release / grasp_retry (see fold_dataset_export.py)",
        "high_level_extra_keys": "pair_index (index into pairs), correction (pick made by the correction loop)",
        "config": config,
    }
