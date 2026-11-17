# 가상·물리 파지 공통 BC → PPO

시연으로 **PPO ActorCriticPolicy의 actor를 BC 사전학습**합니다. 저장한 PPO 체크포인트를
실제 Isaac 환경 평가와 추후 PPO 학습에 그대로 사용합니다. BC에서는 value network를
학습하지 않습니다. SAC는 사용하지 않습니다.

## 정책과 데이터 규약

- 관측 30차원: 모서리 XYZ 12개, 방향 있는 목표 쌍 adjacency matrix 16개,
  virtual/physical one-hot 2개. 모서리 인덱스와 목표 쌍은 에피소드 내에서 유지합니다.
- 행동 4차원: world 좌표 TCP의 `pick_x, pick_y, place_x, place_y`를 [-1,1]로 정규화.
  작업 범위는 X 0.15–0.78m, Y -0.42–0.42m입니다.
- 고수준 한 step은 접근·닫기·들기 확인·이동·열기·후퇴·안정화 전체입니다.
  자세, 깊이, 이동 높이는 실행기가 제어합니다. 파지 자세 자체를 학습하는 저수준 정책은 아닙니다.
- 모드별로 같은 네트워크를 사용하지만, 물리 파지에서는 실제 손가락 접촉/마찰만 사용합니다.
  가상 파지에서만 attachment를 생성합니다.
- 물리 파지는 정점 상승과 TCP 상대 위치 변화로 파지/미끄러짐을 추정합니다.
  접촉 센서에 의한 확정 판정은 아닙니다.

학습 입력은 에피소드별 `high_level.jsonl`, `low_level.npz`, `metadata.json`,
`context.json`입니다. 기존 teleop 수집기가 이 네 파일을 저장합니다.
가상 시연의 기존 pick(천 모서리)은 저수준 구간 첫 프레임의 측정 TCP XY로 변환하고,
물리 시연은 기존 측정 TCP pick을 사용합니다. 첫 프레임은 파지 직후 샘플이므로
정확한 닫기 직전 TCP와 약간 다를 수 있습니다. place는 두 모드 모두 열기 직전 측정 TCP XY입니다.

context에 모드/목표 쌍/좌표 의미가 없거나 세그먼트가 잘못된 데이터는 거부하며,
거부 사유를 `<output>.rejected.json`에 기록합니다. 모드를 폴더 이름으로 추측하지 않습니다.
기존 임시 NPZ와 context 없는 데이터는 이 로더에 바로 넣을 수 없습니다.

## 1. 시연 수집

프로젝트 루트 PowerShell에서 실행합니다.

```powershell
& C:\cloth-folding-rl\scripts\grasp\run_teleop_cloth.ps1 -GraspMode virtual
& C:\cloth-folding-rl\scripts\grasp\run_teleop_cloth.ps1 -GraspMode physical
```

한 번에 하나씩 실행합니다. 조작법은 `README_teleop_cloth.md`를 참고합니다.
성공 episode만 `runs/teleop/<mode>/<episode>/`에 저장됩니다.
각 모드에서 최소 두 에피소드를 준비하면 모드별 train/validation 분리가 가능합니다.
두 개는 연결 점검용 최소량이며, 일반화 평가에 충분한 데이터량은 아닙니다.

## 2. BC 학습

Isaac Sim을 띄우지 않고 실행합니다. Python 환경에 numpy, torch, gymnasium,
stable-baselines3가 필요합니다. 현재 Isaac 번들 환경에는 모두 설치되어 있습니다.

```powershell
Set-Location C:\cloth-folding-rl
& C:\IsaacLab\_isaac_sim\python.bat -m scripts.learning.train_bc --data runs/teleop/virtual runs/teleop/physical --output runs/learning/bc_ppo --epochs 100
```

MLP actor는 128×128이며 정규화한 행동의 MSE로 학습합니다. 검증 coordinate RMSE가
가장 작은 모델을 `bc_ppo.zip`, 학습/검증 에피소드 목록과 기록을 `bc_ppo.json`에 저장합니다.
분리는 에피소드 단위이며 가능한 경우 모드별로 나눕니다. 검증에 쓴 에피소드는 최종
성공률을 보고할 별도 테스트 세트로 사용하지 마세요. 데이터가 없는 모드에서 체크포인트를
실행하려 하면 명시적으로 거부합니다.

## 3. 실제 시뮬레이션 평가

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u scripts/learning/run_policy.py --task evaluate --checkpoint runs/learning/bc_ppo.zip --grasp_mode virtual --episodes 10 --viz kit presets=isaacsim_physx
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u scripts/learning/run_policy.py --task evaluate --checkpoint runs/learning/bc_ppo.zip --grasp_mode physical --episodes 10 --viz kit presets=isaacsim_physx
```

`--viz none`은 화면 없는 실행입니다. `--cloth_x`, `--cloth_y`로 초기 천 위치를 바꿀 수 있습니다.
기본값은 고정 위치 (.50,-.05)m와 회전 0도입니다. seed만 바꾼 반복 실험을 위치/회전
일반화 평가로 해석하면 안 됩니다. 위치별 평가 실행을 별도로 준비하세요.

학습용 시연과 실행기의 이동 궤적/안정화 시간은 다를 수 있습니다. 낮은 BC 오차가 실제
성공을 보장하지 않으므로 모드별 rollout을 확인해야 합니다. 물리 파지 실행기의 고정
접근 자세와 깊이는 시작 설정이며 물리 파지 성공률은 별도 검증 대상입니다.

모든 시도(실패 포함)는 실행별 `runs/learning/<timestamp>/transitions.jsonl`에 남고,
평가는 `evaluation.json`에 성공률을 저장합니다. 성공 기준은 안정화 후 두 모서리 쌍
모두 3D 거리 2cm 이하입니다. 최대 8번 시도 후 timeout으로 truncation 처리하고,
감지한 낙하는 termination 처리합니다. 기존 거리 변화 보상에 파지 실패 -1을 추가합니다.
모서리 거리만으로 천 전체의 형상/평탄도까지 확인할 수는 없습니다.

## 4. 추후 PPO 학습

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u scripts/learning/run_policy.py --task ppo --checkpoint runs/learning/bc_ppo.zip --grasp_mode physical --timesteps 2048 --n_steps 64 --batch_size 32 --viz none presets=isaacsim_physx
```

물리/가상 모드 선택은 실행별 고정입니다. 모드를 섞은 BC는 가능하지만 이 runner에서
PPO를 한 모드로만 계속 학습하면 다른 모드 성능은 떨어질 수 있으니 양쪽을 재평가하세요.
PPO는 on-policy이므로 실패도 해당 rollout 학습에 포함하고, SAC식 replay buffer는 쓰지 않습니다.
BC value network가 학습되지 않은 상태에서 PPO를 시작합니다. 초기 PPO 업데이트가 BC 성능을
떨어뜨릴 수 있으므로 BC 기준 성공률과 비교하세요. `ppo_latest.zip`은 완료/중단 시 저장합니다.
PPO timestep은 **물리 프레임 수가 아니라 전체 Pick & Place 시도 수**입니다. 실제 시뮬레이션은
오래 걸릴 수 있으며, 짧은 연결 검사와 장시간 성능 학습을 구분하세요.

## 확인

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -m unittest discover -s tests -p test_learning_pipeline.py -v
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u scripts/learning/run_policy.py --task smoke --grasp_mode virtual --max_attempts 1 --viz none presets=isaacsim_physx
& C:\IsaacLab\_isaac_sim\python.bat -X utf8 -u scripts/learning/run_policy.py --task smoke --grasp_mode physical --max_attempts 1 --viz none presets=isaacsim_physx
```

단위 검사는 데이터 변환·누출 없는 에피소드 분리·실패/timeout 처리·BC 저장과 PPO 업데이트
연결을 확인합니다. 가짜 backend를 사용한 테스트 결과는 실제 접기 성공률이 아닙니다.
smoke는 체크포인트 없이 기하학적 목표로 실제 한 번 집고 놓는 연결 검사입니다.
명령이 정상 종료되어도 파지 성공 여부와 접기 성공 여부는 출력 결과를 따로 확인하세요.

### 2026-11-17 확인 결과

- 새 파이프라인 단위 검사 7개, 기존 물리 picker 검사 13개 통과.
  BC 체크포인트 저장/로드 및 PPO 32 timestep 업데이트는 가짜 backend에서 확인했습니다.
- 실제 가상 파지 smoke: 약 3.30cm 상승, 한 모서리 이동 후 목표 쌍 거리 약
  1.35cm/28.39cm. 한 번만 옮긴 검사로 전체 접기는 실패이며, 실행/정리 경로는 정상 종료했습니다.
- 실제 물리 파지 smoke: 약 3.27cm 상승 후 운반 중 상대 위치 변화로 미끄러짐 감지.
  실패 보상과 종료를 기록했고 정상 종료했습니다. 물리 접기 성공은 확보하지 못했습니다.
- 현재 `data/demonstrations`의 기존 episode는 `context.json`이 없어 거부됩니다.
  실제 시연 데이터로 학습한 BC 모델이나 실제 환경의 PPO 장시간 학습 결과는 아직 없습니다.
- 실제 실행 기록: `runs/learning/verification/{virtual,physical}/transitions.jsonl`.
