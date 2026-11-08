
import json

import numpy as np

from scripts.learning.trajectory_logger import TrajectoryLogger


def test_demonstration_dataset(tmp_path):
    demo_dir = tmp_path / "episode_001"
    logger = TrajectoryLogger(demo_dir / "high_level.jsonl")

    # 가상의 저수준 데이터 10개
    actions = np.zeros((10, 8), dtype=float)

    # 가상의 고수준 Pick & Place 행동 1개
    logger.add_step(
        observation={"corners": [[0, 0, 0]] * 4},
        action={"pick": [0.1, 0.2], "place": [0.3, 0.2]},
        next_observation={"corners": [[0, 0, 0]] * 4},
        reward=10.0,
        success=True,
        done=True,
    )
    logger.records[-1]["low_level_start"] = 2
    logger.records[-1]["low_level_end"] = 8

    # 성공한 시연에 대해서만 저장
    logger.save()

    np.savez_compressed(
        demo_dir / "low_level.npz",
        action=actions,
    )

    metadata = {
        "episode_id": "episode_001",
        "success": True,
        "high_level_steps": len(logger.records),
        "low_level_steps": len(actions),
        "high_level_file": "high_level.jsonl",
        "low_level_file": "low_level.npz",
    }
    (demo_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    # 세 파일이 모두 존재하는지 확인
    assert (demo_dir / "high_level.jsonl").exists()
    assert (demo_dir / "low_level.npz").exists()
    assert (demo_dir / "metadata.json").exists()

    # 고수준 행동과 저수준 기록이 연결되는지 확인
    record = json.loads(
        (demo_dir / "high_level.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    with np.load(demo_dir / "low_level.npz") as data:
        segment = data["action"][
            record["low_level_start"]:record["low_level_end"]
        ]
        assert len(segment) == 6

    assert json.loads(
        (demo_dir / "metadata.json").read_text(encoding="utf-8")
    )["success"] is True
