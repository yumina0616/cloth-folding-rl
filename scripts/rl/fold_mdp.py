"""Batched (N parallel environments) reward and termination terms for cloth folding.

Same rules as scripts/learning/reward.py, which works on one cloth with corners (4, 3).
Here corners are (N, 4, 3) and every function returns one value per environment.
"""
from __future__ import annotations

import torch

# Same weights as scripts/learning/reward.py so the two agree for one environment
DISTANCE_WEIGHT = 10.0
SUCCESS_BONUS = 10.0
DROP_PENALTY = -5.0
STEP_PENALTY = -0.01


def pair_distances(corners, pairs):
    """Distance of each target corner pair. corners (N, 4, 3), pairs [(i, j), ...] -> (N, P)."""
    # 1. pairs 에서 앞뒤 모서리 인덱스 각각 분리
    first = [i for i,j in pairs]
    second = [j for i,j in pairs]

    # 2. 모든 N개 환경에서 해당 모서리 좌표들을 추출 -> shape: (N, P, 3)
    c1 = corners[:, first, :]
    c2 = corners[:, second, :]

    # 3. 두 모서리 간의 차이 벡터 계산 -> shape: (N, P, 3)
    diff = c1 - c2

    # 4. 거리 계산 -> shape: (N, P)
    distances = torch.linalg.vector_norm(diff, dim=-1)

    return distances


def keypoint_distance(corners, pairs):
    """Mean pair distance per environment -> (N,)."""
    return pair_distances(corners, pairs).mean(dim=-1)


def fold_success(corners, pairs, threshold=0.02):
    """True where every pair is within threshold [m] -> (N,) bool."""
    distances = pair_distances(corners, pairs)
    return (distances <= threshold).all(dim=-1)


def cloth_dropped(nodes, table_z, margin=0.05):
    """True where any cloth vertex fell more than margin below the table top. nodes (N, V, 3) -> (N,) bool."""
    z_coords = nodes[:, :, 2]
    min_z = z_coords.min(dim=-1).values
    return min_z < (table_z - margin)


def fold_reward(previous_corners, current_corners, pairs, success, dropped):
    """Reward per environment -> (N,). success, dropped: (N,) bool."""
    # 1. 이전 평균 거리와 현재 평균 거리 계산
    prev_dist = keypoint_distance(previous_corners, pairs)
    curr_dist = keypoint_distance(current_corners, pairs)

    # 2. 거리 변화량에 따른 보상 -> (N, )
    dist_reward = DISTANCE_WEIGHT * (prev_dist - curr_dist)

    # 3. 성공 보너스 및 떨어짐 패널티 -> (N,)
    success_bonus = SUCCESS_BONUS * success.float()
    drop_penalty = DROP_PENALTY * dropped.float()

    # 4. 모든 항 합치고 스텝 페널티 추가 -> (N,)
    total_reward = dist_reward + success_bonus + drop_penalty + STEP_PENALTY

    return total_reward
