
import json
import tempfile
from pathlib import Path

import torch

from scripts.learning.reward import compute_reward
from scripts.learning.trajectory_logger import TrajectoryLogger


# 직사각형 반 접기
# 0 ─── 1
# │     │
# 2 ─── 3
pairs = [(0, 1), (2, 3)]


def check_success(corners):
    """두 모서리 쌍이 모두 2cm 이내인지 확인"""
    return all(
        torch.linalg.vector_norm(
            corners[i] - corners[j]
        ).item() <= 0.02
        for i, j in pairs
    )


with tempfile.TemporaryDirectory() as temp_dir:

    filepath = Path(temp_dir) / "demo_001.jsonl"
    logger = TrajectoryLogger(filepath)

    # 1. 처음 천의 상태
    corners_0 = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    # 2. 첫 번째 행동: 0번 모서리를 1번 쪽으로 이동
    corners_1 = corners_0.clone()
    corners_1[0, 0] = 0.19

    success_1 = check_success(corners_1)

    reward_1 = compute_reward(
        corners_0,
        corners_1,
        pairs,
        success=success_1
    )

    logger.add_step(
        observation={
            "corners": corners_0.tolist()
        },
        action={
            "pick": [0.0, 0.0],
            "place": [0.2, 0.0],
        },
        next_observation={
            "corners": corners_1.tolist()
        },
        reward=reward_1,
        success=success_1,
        done=False,
    )

    # 3. 첫 번째 행동 이후 상태를 다시 관측
    # 실제 시연에서는 모서리 추적기에서 새 좌표를 읽음

    # 4. 두 번째 행동: 2번 모서리를 3번 쪽으로 이동
    corners_2 = corners_1.clone()
    corners_2[2, 0] = 0.19

    success_2 = check_success(corners_2)

    reward_2 = compute_reward(
        corners_1,
        corners_2,
        pairs,
        success=success_2
    )

    logger.add_step(
        observation={
            "corners": corners_1.tolist()
        },
        action={
            "pick": [0.0, 0.2],
            "place": [0.2, 0.2],
        },
        next_observation={
            "corners": corners_2.tolist()
        },
        reward=reward_2,
        success=success_2,
        done=True,
    )

    # 5. 두 행동의 기록을 JSONL 파일로 저장
    logger.save()

    # 6. 저장된 데이터 확인
    with filepath.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 2

    # 첫 번째 행동: 아직 접기 미완료
    assert records[0]["success"] is False
    assert records[0]["done"] is False

    # 두 번째 행동: 접기 성공 및 종료
    assert records[1]["success"] is True
    assert records[1]["done"] is True

    # 두 번째 행동의 시작 관측은 첫 번째 행동의 결과
    assert (
        records[1]["observation"]["corners"]
        == records[0]["next_observation"]["corners"]
    )

    # 보상도 자동 계산된 값으로 저장됐는지 확인
    assert abs(records[0]["reward"] - reward_1) < 1e-6
    assert abs(records[1]["reward"] - reward_2) < 1e-6

    print("첫 번째 행동 보상:", reward_1)
    print("첫 번째 행동 성공 여부:", success_1)
    print("첫 번째 행동 종료 여부:", records[0]["done"])

    print()

    print("두 번째 행동 보상:", reward_2)
    print("두 번째 행동 성공 여부:", success_2)
    print("두 번째 행동 종료 여부:", records[1]["done"])

    print()
    print("저장된 데이터 개수:", len(records))
    print("두 번의 고수준 행동 기록 테스트 통과!")
