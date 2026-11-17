import json

import numpy as np
import unittest

from scripts.learning.teleop_recorder import TeleopRecorder


def test_manual_segments_and_final_observation(tmp_path):
    recorder = TeleopRecorder(tmp_path / "episode", [(0, 1), (2, 3)])
    before = np.array([[0, 0, 0], [.2, 0, 0], [0, .2, 0], [.2, .2, 0]])
    after = before.copy()
    action = np.array([.5, 0, .2, 1, 0, 0, 0, -1.])
    for index, (source, destination) in enumerate([(0, 1), (2, 3)]):
        recorder.begin_pick_place(after, after[source, :2])
        for step in range(2):
            recorder.append_step(index + step * .1, "carry", action,
                                 np.zeros(9), after, np.zeros(3), True)
        after[source] = after[destination]
        recorder.end_pick_place(after, after[destination, :2])
    # Sensor/control arrays are often reused in place by the simulator.
    action[:] = 99
    after[0, 2] = .005
    assert recorder.finish(after, "episode")
    folder = tmp_path / "episode"
    records = [json.loads(line) for line in (folder / "high_level.jsonl").read_text().splitlines()]
    assert [(r["low_level_start"], r["low_level_end"]) for r in records] == [(0, 2), (2, 4)]
    assert [r["done"] for r in records] == [False, True]
    assert records[-1]["next_observation"]["corners"] == after.tolist()
    with np.load(folder / "low_level.npz", allow_pickle=False) as data:
        assert set(data.files) == {"time_s", "phase", "action", "joint_pos", "corners_w", "tcp_w", "grasping"}
        assert data["action"].shape == (4, 8)
        assert data["action"][0, 0] == .5
        assert data["corners_w"][0, 0, 2] == 0
    assert json.loads((folder / "metadata.json").read_text())["low_level_steps"] == 4
    with unittest.TestCase().assertRaises(RuntimeError):
        recorder.finish(after, "episode")


def _check_discard(tmp_path, aborted):
    recorder = TeleopRecorder(tmp_path / "failed", [(0, 1)])
    corners = np.zeros((4, 3))
    corners[1, 0] = 1
    recorder.begin_pick_place(corners, [0, 0])
    if not aborted:
        recorder.append_step(0, "release", np.zeros(8), np.zeros(9), corners, np.zeros(3), False)
        recorder.end_pick_place(corners, [1, 0])
    assert not recorder.finish(corners, "failed", aborted=aborted)
    assert not (tmp_path / "failed").exists()


def test_incomplete_segment_and_relative_action_rejected(tmp_path):
    recorder = TeleopRecorder(tmp_path / "episode", [(0, 1)])
    corners = np.zeros((4, 3))
    recorder.begin_pick_place(corners, [0, 0])
    with unittest.TestCase().assertRaises(ValueError):
        recorder.end_pick_place(corners, [0, 0])
    with unittest.TestCase().assertRaises(RuntimeError):
        recorder.finish(corners, "episode")
    with unittest.TestCase().assertRaises(ValueError):
        recorder.append_step(0, "carry", np.zeros(7), np.zeros(9), corners, np.zeros(3), True)


def test_failed_episode_creates_no_files(tmp_path):
    _check_discard(tmp_path, False)

def test_aborted_episode_creates_no_files(tmp_path):
    _check_discard(tmp_path, True)
