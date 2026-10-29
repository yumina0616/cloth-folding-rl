
import json
import tempfile
from pathlib import Path

import torch

from scripts.learning.reward import compute_reward
from scripts.learning.trajectory_logger import TrajectoryLogger


# 1. 테스트용 임시 폴더 만들기
with tempfile.TemporaryDirectory() as temp_dir:

    filepath = Path(temp_dir) / "demo_001.jsonl"

    # 2. 기록장 생성
    logger = TrajectoryLogger(filepath)

    # 3. 가상의 접기 전후 모서리 좌표
    previous_corners = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    current_corners = torch.tensor([
        [0.19, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.19, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    # 4. 직사각형 반 접기: 겹쳐야 하는 모서리 쌍
    pairs = [(0, 1), (2, 3)]

    # 5. 접기 성공 여부 확인 (두 쌍 모두 2cm 이하)
    success = all(
        torch.linalg.vector_norm(
            current_corners[i] - current_corners[j]
        ).item() <= 0.02
        for i, j in pairs
    )

    # 6. 보상 함수로 점수 자동 계산
    reward = compute_reward(
        previous_corners,
        current_corners,
        pairs,
        success=success
    )

    # 7. 시연 데이터 기록
    logger.add_step(
        observation={
            "corners": previous_corners.tolist()
        },
        action={
            "pick": [0.0, 0.0, 0.0],
            "place": [0.2, 0.0, 0.0],
        },
        next_observation={
            "corners": current_corners.tolist()
        },
        reward=reward,
        success=success,
        done=True,
    )

    # 8. JSONL 파일 저장
    logger.save()

    # 9. 저장된 파일 확인
    assert filepath.exists()

    with filepath.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 1
    assert records[0]["success"] is True
    assert records[0]["done"] is True
    assert records[0]["action"]["pick"] == [0.0, 0.0, 0.0]
    assert abs(records[0]["reward"] - reward) < 1e-6

    # 10. 결과 출력
    print("저장된 데이터:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))

    print("\n자동 계산된 보상:", reward)
    print("접기 성공 여부:", success)
    print("\n보상 계산 + 데이터 저장 통합 테스트 통과!")
