"""Input/output rules for separate high-level pick and low-level cloth-folding policies.

High-level: 30-D cloth/goal/mode observation -> normalized pick XY (2-D).
Low-level: 28-D cloth/TCP/IK/phase observation -> normalized absolute IK action (8-D).

Coordinate frames:
  High pick XY, cloth corners and TCP: WORLD frame, metres.
  Low action IK position/quaternion: ROBOT ROOT frame.
  Low policy action must be decoded with the SAME bounds used for BC training.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# This is a NEW contract. Legacy 4-D pick+place checkpoints are incompatible.
VERSION = "cloth-hierarchical-v2"
MODES = ("virtual", "physical")
XY_LOW = np.array([.15, -.42], dtype=np.float32)
XY_HIGH = np.array([.78, .42], dtype=np.float32)

HIGH_OBS_SIZE = 30  # corners XYZ (12), goal pairs (16), grasp mode (2)
HIGH_ACTION_SIZE = 2  # world-frame pick XY
LOW_OBS_SIZE = 28  # corners (12), tcp (3), previous IK (8), pair (2), grasp (1), distances (2)
LOW_ACTION_SIZE = 8  # robot-root [x, y, z, qw, qx, qy, qz, gripper]
OBS_SIZE = HIGH_OBS_SIZE  # compatibility with references to old observation size


def finite_array(value, shape):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {shape}, got {result.shape}")
    return result


def encode_observation(corners, pairs, grasp_mode):
    """Legacy 30-D HIGH-LEVEL observation, unchanged."""
    corners = finite_array(corners, (4, 3)).copy()
    if grasp_mode not in MODES:
        raise ValueError(f"Unknown grasp mode: {grasp_mode}")
    goals = np.zeros((4, 4), dtype=np.float32)
    if len(pairs) != 2:
        raise ValueError("Expected two directed source/destination corner pairs")
    indices = [index for pair in pairs for index in pair]
    if sorted(indices) != [0, 1, 2, 3]:
        raise ValueError("Goal pairs must use each corner exactly once")
    for source, destination in pairs:
        goals[source, destination] = 1
    corners[:, :2] = 2 * (corners[:, :2] - XY_LOW) / (XY_HIGH - XY_LOW) - 1
    corners[:, 2] /= .3
    mode = np.array([grasp_mode == name for name in MODES], dtype=np.float32)
    return np.concatenate((corners.ravel(), goals.ravel(), mode))


def normalize_action(world_xy):
    """High-level world-frame pick XY (2,) -> [-1,1] (2,)."""
    xy = finite_array(world_xy, (2,))
    if (xy < XY_LOW).any() or (xy > XY_HIGH).any():
        raise ValueError("Pick TCP XY is outside the policy workspace")
    return (2 * (xy - XY_LOW) / (XY_HIGH - XY_LOW) - 1).astype(np.float32)


def decode_action(action):
    """High-level normalized pick XY (2,) -> world-frame pick XY (2,)."""
    action = finite_array(action, (2,))
    return (XY_LOW + (np.clip(action, -1, 1) + 1) * .5 * (XY_HIGH - XY_LOW)).astype(np.float32)


def encode_low_observation(corners_w, tcp_w, previous_ik_action_root,
                           pair_index, grasping, pair_distances_m):
    """Low-level observation (28,) for one environment.

    Order MUST match the PPO bridge: corners_w(12), tcp_w(3), previous root IK(8),
    pair_index one-hot(2), grasping(1), pair_distances_m(2).
    Use the state observed BEFORE issuing the next policy action.
    """
    corners = finite_array(corners_w, (4, 3))
    tcp = finite_array(tcp_w, (3,))
    ik = finite_array(previous_ik_action_root, (8,))
    distances = finite_array(pair_distances_m, (2,))
    if int(pair_index) != pair_index or int(pair_index) not in (0, 1):
        raise ValueError("pair_index must be 0 or 1")
    if not isinstance(grasping, (bool, np.bool_)):
        raise ValueError("grasping must be a boolean")
    pair = np.eye(2, dtype=np.float32)[int(pair_index)]
    return np.concatenate((corners.ravel(), tcp, ik, pair,
                           np.array([float(grasping)], dtype=np.float32), distances))


def _check_low_bounds(low, high):
    low = finite_array(low, (LOW_ACTION_SIZE,))
    high = finite_array(high, (LOW_ACTION_SIZE,))
    if np.any(high <= low):
        raise ValueError("Each low-level action upper bound must exceed lower bound")
    return low, high


def normalize_low_action(action_root, low, high):
    """Absolute root-frame IK command (8,) -> normalized BC/PPO action (8,).

    Bounds must come from the same training configuration used to save the policy.
    """
    action = finite_array(action_root, (LOW_ACTION_SIZE,))
    low, high = _check_low_bounds(low, high)
    if np.any(action < low) or np.any(action > high):
        raise ValueError("Low-level IK action outside saved bounds")
    if np.linalg.norm(action[3:7]) < 1e-6:
        raise ValueError("Invalid zero IK quaternion")
    return (2 * (action - low) / (high - low) - 1).astype(np.float32)


def decode_low_action(action, low, high):
    """Normalized low policy output (8,) -> robot-root IK command (8,).

    Quaternion is normalized and gripper is discretized for the existing IK runner.
    Workspace and per-step motion safety checks remain the executor's responsibility.
    """
    action = finite_array(action, (LOW_ACTION_SIZE,))
    low, high = _check_low_bounds(low, high)
    decoded = low + .5 * (np.clip(action, -1, 1) + 1) * (high - low)
    magnitude = np.linalg.norm(decoded[3:7])
    if magnitude < 1e-6:
        # Zero quaternion has no rotation; choose identity only as safe fallback.
        decoded[3:7] = [1., 0., 0., 0.]
    else:
        decoded[3:7] /= magnitude
    decoded[7] = 1.0 if decoded[7] >= 0 else -1.0
    return decoded.astype(np.float32)


class _OfflinePolicySpaces(gym.Env):
    """Defines BC/PPO spaces only; not a simulator environment."""
    def reset(self, *, seed=None, options=None):
        raise RuntimeError("Use the Isaac folding environment for rollouts")

    def step(self, action):
        raise RuntimeError("Offline BC space specification cannot execute actions")


class HighPolicySpaces(_OfflinePolicySpaces):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (HIGH_OBS_SIZE,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (HIGH_ACTION_SIZE,), np.float32)


class LowPolicySpaces(_OfflinePolicySpaces):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (LOW_OBS_SIZE,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (LOW_ACTION_SIZE,), np.float32)


# Retain import name for older code, but this is now HIGH-only (2-D action).
PolicySpaces = HighPolicySpaces
