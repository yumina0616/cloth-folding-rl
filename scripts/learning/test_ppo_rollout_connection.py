"""Isaac-free test for the new run_policy.py rollout collector.

Run after placing modified run_policy.py in scripts/learning/.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import torch


def load_collector():
    source = Path(__file__).with_name('run_policy.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'collect_ppo_episode')
    namespace = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[func], type_ignores=[]), '<collector>', 'exec'), namespace)
    return namespace['collect_ppo_episode']


@dataclass
class T:
    observation: object
    action: object
    log_prob: float
    value: float
    reward: float
    done: bool


def fake_sample(model, observation):
    return np.zeros(model.action_size, np.float32), 0.0, -0.5


class FakeExecutor:
    action = torch.zeros((1, 8))


class FakeBridge:
    PICK, CARRY, SETTLE, DONE = 0, 1, 2, 3
    device = 'cpu'

    def __init__(self, failed=False):
        self.ex = FakeExecutor()
        self.phase = torch.tensor([self.PICK])
        self.pair_index = torch.tensor([0])
        self.low_ticks = 0
        self.failed = failed

    def begin_pick(self, xy):
        if self.failed:
            self.phase[0] = self.DONE
            return None, {'pick_initialized': torch.tensor([False])}
        self.phase[0] = self.CARRY
        return None, {'pick_initialized': torch.tensor([True])}

    def step_low_level(self, action):
        if self.phase[0] == self.CARRY:
            self.low_ticks += 1
            self.phase[0] = self.SETTLE
            return None, torch.tensor([2.0]), None, {}
        if self.phase[0] == self.SETTLE:
            self.phase[0] = self.PICK if self.pair_index[0] == 0 else self.DONE
            self.pair_index[0] = 1
            return None, torch.tensor([3.0]), None, {}
        raise AssertionError('unexpected phase')


class FakeModel:
    def __init__(self, size):
        self.action_size = size


def setup_modules(monkeypatch):
    root = types.ModuleType('scripts')
    rl = types.ModuleType('scripts.rl')
    update = types.ModuleType('scripts.rl.ppo_policy_update')
    update.PPOTransition = T
    update.sample_action = fake_sample
    monkeypatch.setitem(sys.modules, 'scripts', root)
    monkeypatch.setitem(sys.modules, 'scripts.rl', rl)
    monkeypatch.setitem(sys.modules, 'scripts.rl.ppo_policy_update', update)


def test_two_options_collect_settling_rewards_and_done(monkeypatch):
    setup_modules(monkeypatch)
    fn = load_collector()
    fn.__globals__['policy_observation'] = lambda b, m, level: np.zeros(30 if level == 'high' else 28)
    fn.__globals__['decode_action'] = lambda raw: np.array([0.5, 0.1])
    fn.__globals__['decode_low_action'] = lambda raw, lo, hi: np.ones(8)
    high, low = fn(FakeBridge(), FakeModel(2), FakeModel(8), 'physical',
                   (np.zeros(8), np.ones(8)), max_iterations=12)
    assert len(high) == 1 and len(high[0]) == 2
    assert [x.reward for x in high[0]] == [5., 5.]
    assert [x.done for x in high[0]] == [False, True]
    assert len(low) == 2 and all(len(ep) == 1 for ep in low)
    assert [ep[0].reward for ep in low] == [5., 5.]
    assert all(ep[-1].done for ep in low)


def test_failed_pick_without_low_samples_is_not_updated(monkeypatch):
    setup_modules(monkeypatch)
    fn = load_collector()
    fn.__globals__['policy_observation'] = lambda b, m, level: np.zeros(30)
    fn.__globals__['decode_action'] = lambda raw: np.array([0.5, 0.1])
    import pytest
    with pytest.raises(RuntimeError, match='No low-level samples'):
        fn(FakeBridge(failed=True), FakeModel(2), FakeModel(8), 'physical',
           (np.zeros(8), np.ones(8)), max_iterations=12)
