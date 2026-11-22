
import json
from pathlib import Path

import numpy as np

from scripts.learning.reward import compute_reward, check_fold_success
from scripts.learning.trajectory_logger import TrajectoryLogger


class DemonstrationRecorder:
    """가상 파지와 물리 파지 시연에서 공통으로 사용하는 기록기."""

    def __init__(self, demo_dir, pairs, success_threshold=0.02):
        self.logger = TrajectoryLogger(
            Path(demo_dir) / "high_level.jsonl"
        )
        self.pairs = pairs
        self.success_threshold = success_threshold

    def record_action(
        self,
        before_corners,
        after_corners,
        pick_xy,
        place_xy,
        low_level_start,
        low_level_end,
    ):
        """Pick & Place 한 번의 결과를 기록한다."""

        success = check_fold_success(
            after_corners,
            self.pairs,
            threshold=self.success_threshold,
        )

        reward = compute_reward(
            before_corners,
            after_corners,
            self.pairs,
            success=success,
        )

        self.logger.add_step(
            observation={
                "corners": before_corners.cpu().tolist()
            },
            action={
                "pick": list(pick_xy),
                "place": list(place_xy),
            },
            next_observation={
                "corners": after_corners.cpu().tolist()
            },
            reward=reward,
            success=success,
            done=False,
        )

        record = self.logger.records[-1]
        record["low_level_start"] = low_level_start
        record["low_level_end"] = low_level_end

        return reward, success

    
    def finalize(self, final_corners, success):
        """시연 종료 후 마지막 행동의 최종 결과를 기록한다."""

        if not self.logger.records:
            raise ValueError("No actions recorded")

        last_record = self.logger.records[-1]

        before_corners = final_corners.new_tensor(
            last_record["observation"]["corners"]
        )

        reward = compute_reward(
            before_corners,
            final_corners,
            self.pairs,
            success=success,
        )

        last_record["next_observation"] = {
            "corners": final_corners.cpu().tolist()
        }
        last_record["reward"] = float(reward)
        last_record["success"] = bool(success)
        last_record["done"] = True


    def save_if_success(self, success, low_level_data, episode_id):
        """성공한 시연의 고수준·저수준 데이터를 함께 저장한다."""

        if not success:
            return False

        if not self.logger.records:
            raise ValueError("No high-level actions recorded")

        if not low_level_data:
            raise ValueError("No low-level data recorded")

        demo_dir = self.logger.filepath.parent
        demo_dir.mkdir(parents=True, exist_ok=True)

        # 고수준 데이터 저장
        self.logger.save()

        # 저수준 데이터 저장
        np.savez_compressed(
            demo_dir / "low_level.npz",
            time_s=np.array([r[0] for r in low_level_data]),
            phase=np.array([r[1] for r in low_level_data]),
            action=np.stack([r[2] for r in low_level_data]),
            joint_pos=np.stack([r[3] for r in low_level_data]),
            corners_w=np.stack([r[4] for r in low_level_data]),
            tcp_w=np.stack([r[5] for r in low_level_data]),
            grasping=np.array([r[6] for r in low_level_data]),
        )

        metadata = {
            "episode_id": episode_id,
            "success": True,
            "high_level_steps": len(self.logger.records),
            "low_level_steps": len(low_level_data),
            "high_level_file": "high_level.jsonl",
            "low_level_file": "low_level.npz",
        }

        (demo_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return True
