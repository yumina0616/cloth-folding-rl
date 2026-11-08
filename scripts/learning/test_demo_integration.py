
"""Test high-level Pick & Place logging without Isaac Sim."""

import json
import tempfile
from pathlib import Path

import torch

from scripts.learning.reward import compute_reward, check_fold_success
from scripts.learning.trajectory_logger import TrajectoryLogger


def test_demo_integration():
    # 1. 가상의 초기 천 모서리 좌표 (4개, 각 x/y/z)
    before_corners = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    pairs = [(0, 1), (2, 3)]

    # 임시 폴더를 사용하므로 실제 시연 데이터 폴더는 건드리지 않음
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "trajectory.jsonl"
        logger = TrajectoryLogger(filepath)

        # 2. 첫 번째 Pick & Place
        source, destination = pairs[0]

        pick_xy = before_corners[source, :2].cpu().tolist()
        place_xy = before_corners[destination, :2].cpu().tolist()

        # Isaac Sim 대신 가상의 접기 결과를 설정
        after_first = before_corners.clone()
        after_first[0, 0] = 0.19

        success1 = check_fold_success(after_first, pairs, threshold=0.02)
        reward1 = compute_reward(
            before_corners, after_first, pairs, success=success1
        )

        logger.add_step(
            observation={"corners": before_corners.cpu().tolist()},
            action={"pick": pick_xy, "place": place_xy},
            next_observation={"corners": after_first.cpu().tolist()},
            reward=reward1,
            success=success1,
            done=False,
        )

        # 3. 두 번째 Pick & Place
        source, destination = pairs[1]

        pick_xy = after_first[source, :2].cpu().tolist()
        place_xy = after_first[destination, :2].cpu().tolist()

        after_second = after_first.clone()
        after_second[2, 0] = 0.19

        success2 = check_fold_success(after_second, pairs, threshold=0.02)
        reward2 = compute_reward(
            after_first, after_second, pairs, success=success2
        )

        logger.add_step(
            observation={"corners": after_first.cpu().tolist()},
            action={"pick": pick_xy, "place": place_xy},
            next_observation={"corners": after_second.cpu().tolist()},
            reward=reward2,
            success=success2,
            done=False,
        )

        # 4. 시연 종료 표시 및 파일 저장
        logger.records[-1]["done"] = True
        logger.save()

        # 5. 저장된 JSONL 파일을 다시 읽어서 검증
        assert filepath.exists()

        with filepath.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 2

        assert records[0]["action"]["pick"] == [0.0, 0.0]
        assert records[0]["success"] is False
        assert records[0]["done"] is False

        assert records[1]["success"] is True
        assert records[1]["done"] is True

        assert records[0]["reward"] > 0
        assert records[1]["reward"] > 10

        for record in records:
            assert len(record["observation"]["corners"]) == 4
            assert len(record["next_observation"]["corners"]) == 4
            assert len(record["action"]["pick"]) == 2
            assert len(record["action"]["place"]) == 2

        print("Integration test passed!")


if __name__ == "__main__":
    test_demo_integration()
