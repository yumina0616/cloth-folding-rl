# Teleoperation 시연 저장 — PR #27 호환

`TeleopRecorder`는 공통 `DemonstrationRecorder`를 호출하는 기록 어댑터입니다.
로봇이나 입력장치를 실행하는 teleoperation 프로그램은 별도로 연결해야 합니다.
자동 접기 스크립트의 실행 방식은 변경하지 않습니다.

## 저장 파일

성공한 episode마다 고유한 폴더를 전달합니다. 같은 폴더를 재사용하면 덮어씁니다.

- `high_level.jsonl`: Pick & Place별 observation/action/next_observation/reward/success/done,
  저수준 구간 `low_level_start`, `low_level_end`.
- `low_level.npz`: `time_s`, `phase`, `action`, `joint_pos`, `corners_w`, `tcp_w`, `grasping`.
- `metadata.json`: episode_id, 성공 여부, 고수준·저수준 개수, 파일명.

저수준 배열은 각각 (T,), (T,), (T,8), (T,9), (T,4,3), (T,3), (T,)입니다.
`action`은 **로봇에 실제 전달한 절대 IK 명령**입니다:
robot root 좌표계의 `[x,y,z,qw,qx,qy,qz,gripper]`, 집게 열기 +1 / 닫기 -1.
키보드의 상대 이동 입력이나 VR 원본 포즈를 직접 넣지 않습니다.
`corners_w`, `tcp_w`는 world 좌표(m), time_s는 episode 시작 기준 초입니다.
모서리 순서와 목표 pairs는 episode 중 유지합니다.
다른 로봇/상대 IK 행동은 별도 변환 또는 명시적인 규격 확장이 필요합니다.

## 연결 위치

프로젝트 루트가 Python import 경로에 있어야 합니다.
아래는 호출 순서를 보여주는 예시이며 독립 실행 프로그램이 아닙니다.

```python
from scripts.learning.teleop_recorder import TeleopRecorder

recorder = TeleopRecorder(episode_dir, pairs, success_threshold=0.02)

# Pick & Place 시도 시작 전: 실제 모서리 상태와 선택한 pick 목표(world XY).
recorder.begin_pick_place(before_corners_w, pick_xy)

# 제어 루프: env.step(action)과 천 상태 갱신 후, 같은 주기로 기록.
recorder.append_step(
    elapsed_s, phase, action[0], robot_joint_pos[0],
    corners_w, measured_tcp_w, actual_grasping,
)

# 놓기 및 안정화 후: 실제 모서리 상태와 의도한 place 목표(world XY).
recorder.end_pick_place(after_corners_w, place_xy)

# 필요한 만큼 Pick & Place 반복 후, 해제·후퇴·최종 안정화.
saved = recorder.finish(final_corners_w, episode_id)
```

`low_level_start:end`는 Python 슬라이스이며 end는 포함하지 않습니다.
고수준 관측은 시도 직전, next_observation은 놓은 뒤의 측정값입니다.
접근도 행동 구간에 포함하려면 접근 전에 begin_pick_place를 호출합니다.
수동 place 목표가 없다면 놓기 직전 측정한 TCP의 XY를 일관되게 사용합니다.
`grasping`은 닫기 키 상태가 아니라 picker의 파지 상태를 사용합니다.

최종 목표 모서리 쌍이 모두 threshold 이내일 때만 저장합니다.
실패 시 저장하지 않으며, 마지막 고수준 행동은 최종 상태와 done=True로 갱신합니다.
중단·리셋 시 `finish(..., aborted=True)`로 폐기하고 다음 episode에 새 인스턴스를 만듭니다.
저장하지 않은 기록은 메모리에만 있으므로 중간 강제 종료에서는 남지 않습니다.
가상/물리 파지 데이터는 구별되는 상위 폴더에 저장하세요.
이 어댑터는 PR #27의 메타데이터 필드를 그대로 사용합니다.
