
import torch

from scripts.learning.reward import (
    compute_reward,
    check_fold_success,
)


# 천의 네 모서리 좌표 (단위: m)
# 0 ─── 1
# │     │
# 2 ─── 3

previous_corners = torch.tensor([
    [0.0, 0.0, 0.0],  # 0: 왼쪽 위
    [0.2, 0.0, 0.0],  # 1: 오른쪽 위
    [0.0, 0.2, 0.0],  # 2: 왼쪽 아래
    [0.2, 0.2, 0.0],  # 3: 오른쪽 아래
])

# 직사각형 반 접기 목표
pairs = [(0, 1), (2, 3)]


# TEST 1: 두 모서리가 목표에 가까워지는 경우
closer_corners = previous_corners.clone()
closer_corners[0, 0] = 0.1
closer_corners[2, 0] = 0.1

reward1 = compute_reward(
    previous_corners, closer_corners, pairs
)

print("TEST 1 - 가까워짐:", reward1)
assert reward1 > 0


# TEST 2: 두 모서리가 목표에서 멀어지는 경우
farther_corners = previous_corners.clone()
farther_corners[0, 0] = -0.1
farther_corners[2, 0] = -0.1

reward2 = compute_reward(
    previous_corners, farther_corners, pairs
)

print("TEST 2 - 멀어짐:", reward2)
assert reward2 < 0


# TEST 3: 두 쌍 모두 2cm 이내로 접은 경우
success_corners = previous_corners.clone()
success_corners[0, 0] = 0.19
success_corners[2, 0] = 0.19

success = check_fold_success(
    success_corners, pairs
)

reward3 = compute_reward(
    previous_corners,
    success_corners,
    pairs,
    success=success
)

print("TEST 3 - 접기 성공:", reward3)
assert success
assert reward3 > 10


# TEST 4: 한 쌍만 목표에 도달한 경우
partial_corners = previous_corners.clone()
partial_corners[0, 0] = 0.19

partial_success = check_fold_success(
    partial_corners, pairs
)

print("TEST 4 - 부분 접기:", partial_success)
assert not partial_success


# TEST 5: 천을 떨어뜨린 경우
reward5 = compute_reward(
    previous_corners,
    previous_corners,
    pairs,
    dropped=True
)

print("TEST 5 - 천 떨어뜨림:", reward5)
assert reward5 < -5


print("\n모든 테스트 통과!")
