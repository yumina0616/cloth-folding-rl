"""Batched fold MDP terms; run without Isaac Sim: python -m pytest tests/test_fold_mdp.py"""
import torch

from scripts.learning.reward import compute_reward
from scripts.rl.fold_mdp import (
    STEP_PENALTY,
    cloth_dropped,
    fold_reward,
    fold_success,
)

# 0 ─── 1
# │     │
# 2 ─── 3      (test_reward.py 와 같은 번호)
FLAT = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]])
PAIRS = [(0, 1), (2, 3)]
TOL = 1e-6


def no(n):
    """(n,) all-False bool tensor, for success/dropped."""
    return torch.zeros(n, dtype=torch.bool)


def closer():
    """test_reward.py TEST 1: corners 0 and 2 move halfway toward 1 and 3."""
    corners = FLAT.clone()
    corners[0, 0] = 0.1
    corners[2, 0] = 0.1
    return corners


def with_pair_gaps(gap_a, gap_b):
    """Corners whose pair (0,1) is gap_a apart and pair (2,3) is gap_b apart, along x."""
    corners = FLAT.clone()
    corners[0, 0] = corners[1, 0] - gap_a
    corners[2, 0] = corners[3, 0] - gap_b
    return corners


def test_reward_shape():
    previous = torch.rand(5, 4, 3)
    current = torch.rand(5, 4, 3)
    reward = fold_reward(previous, current, PAIRS, no(5), no(5))
    assert reward.shape == (5,)


def test_single_env_matches_compute_reward():
    batched = fold_reward(FLAT[None], closer()[None], PAIRS, no(1), no(1))
    single = compute_reward(FLAT, closer(), PAIRS)
    assert abs(batched[0].item() - single) < TOL


def test_success_and_drop_terms_match_compute_reward():
    batched = fold_reward(
        FLAT[None], closer()[None], PAIRS,
        torch.tensor([True]), torch.tensor([True]),
    )
    single = compute_reward(FLAT, closer(), PAIRS, success=True, dropped=True)
    assert abs(batched[0].item() - single) < TOL


def test_envs_are_independent():
    previous = torch.stack([FLAT, FLAT])
    current = torch.stack([closer(), FLAT])
    reward = fold_reward(previous, current, PAIRS, no(2), no(2))
    assert reward[0] > 0
    assert abs(reward[1].item() - STEP_PENALTY) < TOL


def test_fold_success_needs_every_pair():
    corners = torch.stack([
        with_pair_gaps(0.01, 0.01),  # 두 쌍 다 기준(0.02) 안 -> 성공
        with_pair_gaps(0.0, 0.04),   # 한 쌍만 붙음 -> 실패
    ])
    assert fold_success(corners, PAIRS).tolist() == [True, False]


def test_cloth_dropped():
    # env 0: corner carried high          -> not dropped
    # env 1: corner on table, hand up     -> dropped
    # env 2: corner and hand both low     -> end of fold, not dropped
    # env 3: same as env 1 but not grasping -> nothing to drop
    carried_z = torch.tensor([0.10, 0.005, 0.005, 0.005])
    hand_z = torch.tensor([0.15, 0.15, 0.03, 0.15])
    grasping = torch.tensor([True, True, True, False])
    assert cloth_dropped(carried_z, hand_z, grasping).tolist() == [False, True, False, False]


def test_cloth_dropped_uses_table_height():
    # Same heights as env 1 above, shifted by a table at z = 0.5
    dropped = cloth_dropped(torch.tensor([0.505]), torch.tensor([0.65]), torch.tensor([True]), table_z=0.5)
    assert dropped.tolist() == [True]
