
# scripts/learning/reward.py

from scripts.metrics.fold_metrics import keypoint_distance


def compute_reward(
    previous_corners,
    current_corners,
    pairs,
    success=False,
    dropped=False,
):
    """
    천 접기 행동 이후의 보상을 계산한다.

    previous_corners: 행동 전 모서리 좌표 (4, 3)
    current_corners: 행동 후 모서리 좌표 (4, 3)
    pairs: 겹쳐야 할 모서리 인덱스 쌍
    success: 이번 행동으로 접기 성공 여부
    dropped: 천 떨어뜨림 여부
    """

    # 1. 이전 모서리 거리
    previous_distance = keypoint_distance(
        previous_corners, pairs
    )

    # 2. 현재 모서리 거리
    current_distance = keypoint_distance(
        current_corners, pairs
    )

    # 3. 모서리 거리 변화에 따른 보상
    distance_reward = 10.0 * (
        previous_distance - current_distance
    )

    # 4. 접기 성공 보너스
    success_reward = 10.0 if success else 0.0

    # 5. 천 떨어뜨림 패널티
    drop_penalty = -5.0 if dropped else 0.0

    # 6. 스텝 패널티
    step_penalty = -0.01

    # 7. 최종 보상
    total_reward = (
        distance_reward
        + success_reward
        + drop_penalty
        + step_penalty
    )

    return total_reward

def check_fold_success(corners, pairs, threshold=0.02):
    """모든 목표 모서리 쌍이 2cm 이내인지 확인한다."""
    import torch

    for i, j in pairs:
        distance = torch.linalg.vector_norm(
            corners[i] - corners[j]
        ).item()

        if distance > threshold:
            return False

    return True