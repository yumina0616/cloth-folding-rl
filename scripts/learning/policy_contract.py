"""Shared high-level policy contract for virtual and physical pick/place."""
import gymnasium as gym
import numpy as np

VERSION = "cloth-pick-place-v1"
MODES = ("virtual", "physical")
XY_LOW = np.array([.15, -.42], dtype=np.float32)
XY_HIGH = np.array([.78, .42], dtype=np.float32)
OBS_SIZE = 30  # corners XYZ (12), directed goal pairs (16), grasp mode (2)


def finite_array(value, shape):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {shape}, got {result.shape}")
    return result


def encode_observation(corners, pairs, grasp_mode):
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
    xy = finite_array(world_xy, (4,)).reshape(2, 2)
    if (xy < XY_LOW).any() or (xy > XY_HIGH).any():
        raise ValueError("Pick/place TCP XY is outside the policy workspace")
    return (2 * (xy - XY_LOW) / (XY_HIGH - XY_LOW) - 1).ravel()


def decode_action(action):
    action = finite_array(action, (4,))
    return (XY_LOW + (np.clip(action, -1, 1).reshape(2, 2) + 1) * .5 * (XY_HIGH - XY_LOW)).ravel()


class PolicySpaces(gym.Env):
    """Space specification for offline BC, never a simulated training environment."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (4,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("Use the Isaac pick/place environment for rollouts")

    def step(self, action):
        raise RuntimeError("Offline BC space specification cannot execute actions")
