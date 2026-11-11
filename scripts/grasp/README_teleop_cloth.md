# 키보드로 천 접기와 시연 저장

설치된 Isaac Lab 3.0 / Isaac Sim PhysX 환경용입니다.
Franka와 30cm 천, 기존 PhysX 단일 정점 가상 파지를 사용합니다.
`--grasp_mode physical`로 접촉·마찰 기반 물리 파지도 선택할 수 있습니다.
영상 녹화·강화학습 실행은 포함하지 않습니다.

## 물리 파지

PowerShell 스크립트 실행이 차단된 환경에서는 아래 명령을 그대로 실행합니다.

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u C:\cloth-folding-rl\scripts\grasp\teleop_cloth.py --grasp_mode physical --viz kit presets=isaacsim_physx
```

가상 연결과 손·천 충돌 제외를 사용하지 않고 손가락 마찰로 잡습니다.
집게 열림 폭은 기본 4cm, 손가락 마찰은 static 3.0 / dynamic 2.0, 힘 한도는 70N입니다.
마찰을 이전 2.0 / 1.5보다 높인 비교용 설정이며, 파지 개선 효과는 아직 검증하지 않았습니다.
모서리 자체보다 조금 안쪽의 `PHYSICAL` 안내에 표시되는 pinch point로 이동하세요.
초기 평면 메시에서 선택한 정점 번호를 계속 추적합니다.
K는 집게 열기·닫기만 수행합니다. 천이 멀리 있어도 닫기 명령은 실행됩니다.
자동 들기·파지 실패 시 강제 열기·해제 후 강제 대기를 수행하지 않습니다.
이동 키로 직접 내려가고, 들어 올리고, 옮기고, 놓습니다.
손가락을 닫은 직전부터 열기 직후의 첫 제어 스텝까지를 한 행동으로 기록합니다.
물리 파지는 모서리에 한정하지 않으며, 닫는 순간 가까운 천 정점으로 파지 상태를 추정합니다.
추정은 천/TCP 상승량과 상대 위치 변화에 기반하며 조작이나 저장 자격을 바꾸지 않습니다.
천이 들어 올려지기 전이나 추적할 정점이 없을 때의 grasping=False는 확정적인 파지 실패를 뜻하지 않습니다.
모서리를 잡지 못한 시도가 있어도 재시도할 수 있고, P를 눌렀을 때 최종 접기 상태로 성공을 판정합니다.
성공 데이터 기본 저장 위치는 `runs/teleop/physical/<episode_id>/`입니다.
PR #27의 세 파일 규격은 같고, context.json에 파지 모드와 검증 결과를 기록합니다.

물리 파지 통합 검사에는 `--grasp_mode physical --smoke_test --viz none`을 사용합니다.
수동 명령을 대신하는 테스트 입력으로 닫기·들기·열기·최종 실패 저장 차단을 검사합니다.
실제 천이 들리지 않아도 조작 차단이나 자동 열기가 없는지 확인하며, 들기 성공으로 보고하지 않습니다.

## 실행

```powershell
& C:\cloth-folding-rl\scripts\grasp\run_teleop_cloth.ps1
```

느리게 이동하려면:

```powershell
& C:\cloth-folding-rl\scripts\grasp\run_teleop_cloth.ps1 -Speed 0.03
```

직접 실행:

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u C:\cloth-folding-rl\scripts\grasp\teleop_cloth.py --viz kit presets=isaacsim_physx
```

창이 뜬 뒤 콘솔의 `[teleop] READY` 안내를 기다리고 뷰포트를 클릭합니다.
안정화와 손목 정렬 중에는 입력을 받지 않습니다.

| 키 | 동작 |
|---|---|
| W / S | world X 증가 / 감소 |
| A / D | world Y 증가 / 감소 |
| Q / E | world Z 증가 / 감소 |
| K | 모서리 잡기 / 놓기 |
| P | 시연 종료: 안정화 후 성공 판정 및 저장 |
| R | 현재 기록 폐기, 로봇·천 초기화, 새 episode |
| Esc | 종료, 미저장 시연 폐기 |

손끝 자세는 아래를 향하도록 고정합니다. 기본 속도는 시뮬레이션 시간 기준 6cm/s입니다.
목표 이동 범위는 X 0.15–0.78m, Y -0.42–0.42m, Z 0.004–0.60m입니다.
실제 손끝과 목표가 8cm 이상 벌어지면 추가 목표 이동을 제한합니다.
R을 누르기 전 이동 키를 놓습니다. 창을 전환할 때도 이동 키를 놓습니다.

## 접기 순서

1. 가까운 모서리 위로 이동한 뒤 내려갑니다.
2. 가상 파지에서는 K를 누릅니다. 실제 손끝에서 가장 가까운 **추적 모서리**가 2.5cm 이내여야 잡힙니다.
   실패하면 `GRASP MISSED`와 거리가 출력되며 행동 기록은 시작하지 않습니다.
3. Q로 들어 올려 반대편 대응 모서리로 옮긴 뒤 내려갑니다.
4. K로 놓습니다. 가상 파지에서는 연결을 해제한 뒤 2초 안정화합니다.
   가상 파지의 이 구간에서는 이동 입력을 무시합니다. `RELEASED`가 뜨면 다시 움직일 수 있습니다.
5. 다른 모서리도 반복합니다. 필요하면 다시 잡아 보정할 수 있습니다.
6. 손끝을 후퇴시키고 P를 누릅니다. 2초간 정지한 뒤 성공 여부를 판정합니다.
7. `SAVED` 또는 `FAILED`를 확인하고 R로 다음 시연을 시작합니다.

집은 상태이거나 아직 완료한 행동이 없으면 P는 종료하지 않습니다.
두 목표 모서리 쌍의 최종 3D 거리가 모두 2cm 이하일 때만 성공으로 저장합니다.
이는 모서리 기준 판정이며 천 전체 평탄도까지 평가하지 않습니다.
기본 600초를 넘긴 episode는 폐기·초기화합니다.

## 데이터

기본 폴더: `C:\cloth-folding-rl\runs\teleop\virtual\<episode_id>\`

- `high_level.jsonl`, `low_level.npz`, `metadata.json`: PR #27 공통 규격.
- `context.json`: teleoperation/가상 파지 출처, 좌표계, 모서리 정점·목표 쌍, 샘플 간격, 실행 옵션.

가상 파지의 고수준 행동은 파지 시도 직전부터 해제 후 안정화까지입니다.
물리 파지는 닫기 직전부터 수동 열기 직후까지입니다.
접근·행동 사이의 이동·마지막 후퇴는 저수준 기록에 포함되지만 행동 구간 밖에 있을 수 있습니다.
pick은 가상 파지에서는 천 모서리, 물리 파지에서는 닫기 직전 실제 TCP의 world XY입니다.
place는 열기 직전 실제 TCP의 world XY입니다.
저수준 action은 매 물리 스텝에 실제 전달한 root 좌표계 절대 IK 8차원 명령입니다.
최종 안정화 상태를 마지막 고수준 next_observation에 반영하고 done=True로 저장합니다.
실패·리셋·중단한 episode는 데이터 폴더를 만들지 않습니다.

## 자동 통합 검사

```powershell
& C:\cloth-folding-rl\scripts\grasp\run_teleop_cloth.ps1 -SmokeTest
```

실제 천 접근·파지·6cm 상승·해제·실패 저장 차단·평평한 천 복원을 검사하고 종료합니다.
키보드 입력은 사용하지 않으며 성공한 접기 데이터를 만들어내지 않습니다.
데이터 규격 테스트는 `scripts/learning/test_teleop_recorder.py`에서 별도로 수행합니다.
