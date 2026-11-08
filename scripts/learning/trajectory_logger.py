
import json
from pathlib import Path


class TrajectoryLogger:
    def __init__(self, filepath):
        # 데이터를 저장할 파일 경로
        self.filepath = Path(filepath)

        # 시연 기록을 임시로 보관할 리스트
        self.records = []

    def add_step(
        self,
        observation,
        action,
        next_observation,
        reward,
        success,
        done=False,
    ):
        # 시연 한 번의 데이터를 기록
        record = {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
            "reward": float(reward),
            "success": bool(success),
            "done": bool(done),
        }

        self.records.append(record)

    def save(self):
        self.filepath.parent.mkdir(
            parents=True, exist_ok=True
        )

        with self.filepath.open(
            "w", encoding="utf-8"
        ) as f:
            for record in self.records:
                f.write(
                    json.dumps(record) + "\n"
                )
