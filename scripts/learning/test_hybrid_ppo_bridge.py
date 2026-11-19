"""Stage 7 integration contract tests; run with pytest -q.

Uses a mock executor (no Isaac Sim needed). Verifies reward and phase transitions;
NOT a simulation validation or a PPO-training test.
"""
from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch

# Stubs the MDP implementation if this test is run without repository installed.
# In the repository, import the actual fold_mdp.py from scripts/rl.
ROOT = Path(__file__).resolve().parents[2]
FALLBACK = Path(__file__).resolve().parents[0]
MDP_PATH = (ROOT / "scripts" / "rl" / "fold_mdp.py")
if not MDP_PATH.exists():
    # Tests can also be run next to copied stage-7 files.
    MDP_PATH = FALLBACK / "fold_mdp.py"

def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

mdp = import_from_path("scripts.rl.fold_mdp", MDP_PATH)
bridge_module = import_from_path("hybrid_ppo_bridge_under_test", Path(__file__).resolve().parents[1] / "rl" / "hybrid_ppo_bridge.py")
HybridPPOBridge = bridge_module.HybridPPOBridge


class MockExecutor:
    def __init__(self, n=2):
        self.num_envs = n
        self.device = 'cpu'
        self.physical = True
        self.base = types.SimpleNamespace(physics_dt=0.5)
        self.action = torch.zeros((n, 8))
        self.action[:, 3] = 1.0
        self.action[:, 7] = -1.0
        self.held_vertex = torch.full((n,), -1, dtype=torch.long)
        self.corners_tensor = torch.tensor(
            [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]]
        ).repeat(n, 1, 1)
        self.calls = 0
        self.pick_calls = 0
        self.next_release = torch.zeros(n, dtype=torch.bool)
        self.next_drop = torch.zeros(n, dtype=torch.bool)
        self.next_slip = torch.zeros(n, dtype=torch.bool)
        self.lift_verified = torch.ones(n, dtype=torch.bool)

    def corners(self):
        return self.corners_tensor.clone()

    def tcp(self):
        return torch.zeros((self.num_envs, 3))

    def grasp_target(self, source):
        return self.corners_tensor[torch.arange(self.num_envs), source]

    def grasp_corner(self, source, active=None, pick_xy=None, policy_controls_lift=False):
        assert policy_controls_lift and pick_xy.shape == (self.num_envs, 2)
        self.pick_calls += 1
        self.held_vertex[active] = source[active]
        return active.clone(), {'mock': True}

    def step_policy_action(self, action, active=None):
        assert torch.allclose(action[:, 3:7].norm(dim=1), torch.ones(self.num_envs))
        self.calls += 1
        self.action[active] = action[active]
        released = active & self.next_release
        self.held_vertex[released] = -1
        return {'released': released, 'dropped': active & self.next_drop,
                'slipped': active & self.next_slip, 'lift_verified': self.lift_verified.clone()}


def make_bridge():
    ex = MockExecutor()
    return ex, HybridPPOBridge(ex, [(0, 1), (2, 3)], settle_seconds=1.0, max_low_steps=4)


def valid_action(ex):
    return ex.action.clone()


def test_two_picks_and_two_releases_reach_done():
    ex, b = make_bridge()
    obs, info = b.begin_pick(torch.zeros((2, 2)))
    assert info['pick_initialized'].all()
    assert (b.phase == b.CARRY).all()
    ex.next_release[:] = True
    obs, rew, done, info = b.step_low_level(valid_action(ex))
    assert (b.phase == b.SETTLE).all()
    ex.next_release[:] = False
    # Settling must not require a policy quaternion; all-zero action is allowed.
    obs, rew, done, info = b.step_low_level(torch.zeros((2, 8)))
    assert (b.phase == b.PICK).all() and (b.pair_index == 1).all()
    b.begin_pick(torch.zeros((2, 2)))
    ex.next_release[:] = True
    b.step_low_level(valid_action(ex))
    ex.next_release[:] = False
    obs, rew, done, info = b.step_low_level(torch.zeros((2, 8)))
    assert done.all() and (b.phase == b.DONE).all()
    assert not info['fold_success'].any()  # corners did not actually move
    assert ex.pick_calls == 2


def test_drop_causes_failure():
    ex, b = make_bridge()
    b.begin_pick(torch.zeros((2, 2)))
    ex.next_drop[0] = True
    obs, reward, done, info = b.step_low_level(valid_action(ex))
    assert bool(done[0]) and not bool(done[1])
    assert bool(info['invalid'][0])
    assert reward[0] < reward[1]


def test_timeout_and_input_shape():
    ex, b = make_bridge()
    with pytest.raises(ValueError):
        b.begin_pick(torch.zeros((2, 3)))
    b.begin_pick(torch.zeros((2, 2)))
    with pytest.raises(ValueError):
        b.step_low_level(torch.zeros((2, 7)))
    for _ in range(4):
        obs, reward, done, info = b.step_low_level(valid_action(ex))
    assert done.all() and info['timeout'].all()


def test_success_bonus_on_final_settle_only():
    ex, b = make_bridge()
    b.begin_pick(torch.zeros((2, 2)))
    ex.next_release[:] = True
    b.step_low_level(valid_action(ex))
    ex.next_release[:] = False
    b.step_low_level(torch.zeros((2, 8)))
    b.begin_pick(torch.zeros((2, 2)))
    # Make all paired corners coincide at the end of the second placement.
    ex.corners_tensor[:, 1] = ex.corners_tensor[:, 0]
    ex.corners_tensor[:, 3] = ex.corners_tensor[:, 2]
    ex.next_release[:] = True
    _, rew1, _, info1 = b.step_low_level(valid_action(ex))
    assert not info1['fold_success'].any()
    ex.next_release[:] = False
    _, rew2, done, info2 = b.step_low_level(torch.zeros((2, 8)))
    assert done.all() and info2['fold_success'].all()
    assert (rew2 >= 9).all()
