"""BC policy output <-> hybrid RL bridge interface tests (no Isaac Sim).

Run at repository root:
    python -m pytest -q scripts/learning/test_bc_rl_integration.py
"""
import numpy as np
import torch

from scripts.learning.policy_contract import (
    HIGH_ACTION_SIZE, HIGH_OBS_SIZE, LOW_ACTION_SIZE, LOW_OBS_SIZE,
    decode_action, decode_low_action, encode_observation,
    encode_low_observation, normalize_action, normalize_low_action,
)
from scripts.rl.hybrid_ppo_bridge import HybridPPOBridge

PAIRS = [(0, 1), (2, 3)]
CORNERS = np.array([
    [0.35, -0.20, 0.0],
    [0.65, -0.20, 0.0],
    [0.35, 0.10, 0.0],
    [0.65, 0.10, 0.0],
], dtype=np.float32)


def test_high_pick_matches_bridge_world_xy():
    obs = encode_observation(CORNERS, PAIRS, "physical")
    pick = np.array([0.36, -0.19], dtype=np.float32)
    prediction = normalize_action(pick)
    recovered = decode_action(prediction)
    assert obs.shape == (HIGH_OBS_SIZE,)
    assert prediction.shape == (HIGH_ACTION_SIZE,) == (2,)
    np.testing.assert_allclose(recovered, pick, atol=1e-6)


def test_low_action_uses_same_root_frame_and_eight_dimensions():
    previous_ik = np.array([0.3, -0.1, 0.15, 1, 0, 0, 0, -1], dtype=np.float32)
    next_ik = np.array([0.31, -0.1, 0.16, 1, 0, 0, 0, -1], dtype=np.float32)
    lower = np.array([0.1, -0.4, 0.0, -1, -1, -1, -1, -1], dtype=np.float32)
    upper = np.array([0.8, 0.4, 0.5, 1, 1, 1, 1, 1], dtype=np.float32)
    pair_distances = np.array([0.3, 0.3], dtype=np.float32)
    obs = encode_low_observation(CORNERS, np.array([0.3, -0.1, 0.15]),
                                 previous_ik, 0, True, pair_distances)
    encoded = normalize_low_action(next_ik, lower, upper)
    decoded = decode_low_action(encoded, lower, upper)
    assert obs.shape == (LOW_OBS_SIZE,)
    assert encoded.shape == (LOW_ACTION_SIZE,) == (8,)
    np.testing.assert_allclose(decoded, next_ik, atol=1e-6)


class _FakeExecutor:
    """The minimum BatchedFoldExecutor interface used by HybridPPOBridge."""
    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.base = type("Base", (), {"physics_dt": 0.5})()
        self.physical = True
        self.action = torch.tensor([[0.3, -0.1, 0.15, 1., 0., 0., 0., -1.]])
        self.held_vertex = torch.tensor([-1])
        self.positions = torch.tensor(CORNERS[None, :, :].copy())
        self.pick_xys = []
        self.commands = []

    def corners(self):
        return self.positions.clone()

    def tcp(self):
        return torch.tensor([[0.3, -0.1, 0.15]])

    def grasp_target(self, source):
        return self.positions[torch.arange(1), source]

    def grasp_corner(self, source, active=None, pick_xy=None, policy_controls_lift=False):
        assert policy_controls_lift
        assert active.item()
        self.pick_xys.append(pick_xy.clone())
        self.held_vertex[0] = source[0]
        return active.clone(), {}

    def step_policy_action(self, action, active=None):
        self.commands.append(action.clone())
        released = active & (action[:, 7] > 0)
        self.action[active] = action[active]
        if released.item():
            self.held_vertex[0] = -1
        return {
            "released": released,
            "dropped": torch.zeros(1, dtype=torch.bool),
            "slipped": torch.zeros(1, dtype=torch.bool),
            "lift_verified": torch.ones(1, dtype=torch.bool),
        }


def test_encoded_bc_actions_reach_bridge_without_dimension_conversion():
    ex = _FakeExecutor()
    bridge = HybridPPOBridge(ex, PAIRS, settle_seconds=0.5, max_low_steps=10)
    original_pick = np.array([0.36, -0.19], dtype=np.float32)
    bc_pick_output = normalize_action(original_pick)
    pick_world = decode_action(bc_pick_output)
    bridge.begin_pick(torch.tensor(pick_world[None, :]))
    np.testing.assert_allclose(ex.pick_xys[0].numpy()[0], original_pick, atol=1e-6)

    lower = np.array([0.1, -0.4, 0.0, -1, -1, -1, -1, -1], dtype=np.float32)
    upper = np.array([0.8, 0.4, 0.5, 1, 1, 1, 1, 1], dtype=np.float32)
    release_ik = np.array([0.31, -0.1, 0.16, 1, 0, 0, 0, 1], dtype=np.float32)
    normalized = normalize_low_action(release_ik, lower, upper)
    decoded = decode_low_action(normalized, lower, upper)
    _, _, _, info = bridge.step_low_level(torch.tensor(decoded[None, :]))
    assert ex.commands[0].shape == (1, 8)
    assert info["released"].item()
    assert bridge.phase.item() == bridge.PICK  # settling completed
    assert bridge.pair_index.item() == 1
