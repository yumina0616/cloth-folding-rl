
import json

import numpy as np
import torch

from scripts.learning.demonstration_recorder import DemonstrationRecorder


def test_demonstration_recorder(tmp_path):
    demo_dir = tmp_path / "episode_001"
    pairs = [(0, 1), (2, 3)]

    recorder = DemonstrationRecorder(demo_dir, pairs)

    # 접기 전 모서리 좌표
    before = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    # 접기 후: 두 모서리 쌍이 모두 2cm 이내
    after = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.01, 0.2, 0.0],
    ])

    reward, success = recorder.record_action(
        before_corners=before,
        after_corners=after,
        pick_xy=(0.0, 0.0),
        place_xy=(0.2, 0.0),
        low_level_start=0,
        low_level_end=2,
    )

    assert success is True
    assert reward > 10

    # 시연 종료 시 마지막 행동의 결과가 갱신되는지 확인
    recorder.finalize(
        final_corners=after,
        success=True,
    )

    last_record = recorder.logger.records[-1]

    assert last_record["done"] is True
    assert last_record["success"] is True
    assert last_record["next_observation"]["corners"] == after.tolist()

    # #19와 동일한 7개 항목 구조의 가상 저수준 데이터
    low_level_data = [
        (
            i * 0.01,
            "fold",
            np.zeros(8),
            np.zeros(9),
            np.zeros((4, 3)),
            np.zeros(3),
            False,
        )
        for i in range(2)
    ]

    saved = recorder.save_if_success(
        success=True,
        low_level_data=low_level_data,
        episode_id="episode_001",
    )

    assert saved is True
    assert (demo_dir / "high_level.jsonl").exists()
    assert (demo_dir / "low_level.npz").exists()
    assert (demo_dir / "metadata.json").exists()

    record = json.loads(
        (demo_dir / "high_level.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert record["low_level_start"] == 0
    assert record["low_level_end"] == 2

    with np.load(demo_dir / "low_level.npz") as data:
        assert len(data["action"]) == 2

    # 실패한 시연은 데이터 폴더를 생성하지 않는지 확인
    failed_dir = tmp_path / "failed_episode"
    failed_recorder = DemonstrationRecorder(failed_dir, pairs)

    assert failed_recorder.save_if_success(
        success=False,
        low_level_data=low_level_data,
        episode_id="failed_episode",
    ) is False

    assert not failed_dir.exists()
