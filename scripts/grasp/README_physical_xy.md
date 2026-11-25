# 물리 파지 XY 범위 탐색: pinch-closer-corner 기준

기준 브랜치는 `exp/pinch-closer-corner`입니다. 해당 브랜치의 물리 파지, IoU 측정,
task_config_compat, BC exporter를 사용합니다. 이전 `exp/grasp-gripper-yaw`에서 만든
탐색 결과와 합치지 않습니다. 기존 물리 파지 동작은 변경하지 않습니다.

## 기본 설정

- 천 중심 (0.50,-0.05)m, 30cm 천, 회전 0도, 추가 위치 jitter 0.
- 각 축 이동 -3cm/0/+3cm: 9개 위치 × 위치별 3회 = 총 27회.
- 최초 기준 위치 3회를 먼저 수행하고 전부 실패하면 다른 위치는 실행하지 않습니다.
- 물리 파지, 전체 접기(`stop_after=fold`), 파지 재시도 1회.
- 스크린샷의 추천 설정을 `probe_physical_xy.py`의 PROFILE에 명시합니다:

```text
--success_distance 0.05 --pinch_jaw_shift 0.01
--close_lift 0.002 --check_lift 0.02 --arc_peak 0.04
--table_friction 0.6 --table_static_friction 0.65
--release_height 0.04 --place_inset 0.01 --move_axis y
--free_speed 2 --retry_in_place --place_settle_s 0.5
```

스크린샷은 `--shape_iou_threshold 0`으로 IoU 판정 문턱을 끈 테스트였습니다.
이 탐색기의 기본 수집 판정은 **두 모서리 쌍 각각 5cm 이하 AND IoU ≥0.8**입니다.
이전 테스트 명령과 같은 IoU 설정은 `--iou_min 0`으로 명시합니다.
기존 metric의 top-view IoU를 사용하며 높이 방향 주름/전체 평탄도까지 판정하지 않습니다.
`--min_rate .8`은 위치별 반복 성공률 기준으로, IoU 문턱과 별개입니다.
3회 반복에서 통과 후보가 되려면 3회 모두 성공해야 합니다. 이는 연속 영역이나 통계적
신뢰도를 입증하는 값이 아니며 통과 위치에서 더 많은 반복과 중간 위치 확인이 필요합니다.

## 실행

이 패치 브랜치가 있는 저장소 루트에서 실행하세요. 작성 시 별도 작업 폴더는 아래와 같습니다.

```powershell
Set-Location C:\cloth-folding-rl\runs\physical-xy-patch

# 먼저 계획만 출력: 시뮬레이터를 실행하지 않습니다.
& C:\IsaacLab\_isaac_sim\python.bat scripts/grasp/probe_physical_xy.py --video

# 기준 위치 한 번만 실행, 성공/실패 모두 영상 저장.
& C:\IsaacLab\_isaac_sim\python.bat scripts/grasp/probe_physical_xy.py --run --video --max_trials 1

# 같은 계획을 이어서 실행. 완료한 시도는 건너뜁니다.
& C:\IsaacLab\_isaac_sim\python.bat scripts/grasp/probe_physical_xy.py --run --video
```

스크린샷의 IoU 문턱 0 설정을 재현할 때는 별도 출력 폴더를 사용합니다.

```powershell
& C:\IsaacLab\_isaac_sim\python.bat scripts/grasp/probe_physical_xy.py --run --video --iou_min 0 --max_trials 1 --out runs/xy-reference-iou0
```

카메라/인코더가 정상 동작하는지 먼저 1회 확인한 뒤 전체 실행을 시작하세요.
영상 옵션은 기본적으로 꺼져 있으며, `--video`를 켜면 실패한 시도도 저장합니다.
15초마다 경과 시간과 최근 로그를 콘솔에 출력합니다. 실행 오류/잘못된 RESULT/영상 누락 시
중단하고 원인을 확인하도록 합니다. 코드나 프로필이 달라지면 새 `--out`을 사용합니다.

## 결과

기본 `runs/physical-xy-probe/` 아래에 저장합니다.

- `plan.json`: 전체 실행 명령의 프로필·좌표·seed와 전문가/지표 코드 fingerprint.
- `results.jsonl`: 시도별 성공 여부, 거리, IoU, corner/shape 성공, 파지 기록, 영상 경로.
- `summary.csv/json`: 위치별 성공률, 평균 IoU, 오류 수와 통과 후보 여부.
- `logs/<trial>.log`: 원본 시뮬레이터 로그.
- `artifacts/<trial>/<timestamp>/`: MP4, result.json, 설정, 진단 및 기존 BC exporter 출력.

전체 접기 전에 파지 실패로 종료된 실행은 IoU가 없을 수 있으며 이 경우 실패로 기록합니다.
전체 접기를 완료했다고 보고했는데 IoU가 없다면 잘못된 코드/결과로 보고 중단합니다.
기존 exporter의 high_level.jsonl/low_level.npz 등은 upstream 규약입니다. 이 패치는
명령 직전 관측을 쓰는 별도 `low_level_bc.npz` 형식으로 변환하지 않습니다.

## 검증과 제한

```powershell
& C:\IsaacLab\_isaac_sim\python.bat -m unittest discover -s tests -p test_probe_physical_xy.py -v
```

12개 검사로 물리/전체 접기 명령, 기본 프로필, 위치 검증, IoU 누락/문턱 검사,
영상 옵션, 중단/재개와 기준 위치 실패 중단을 확인했습니다. 실제 전체 접기와 MP4 출력은
이 새 패치에서 아직 재검증하지 않았습니다. 이전 브랜치에서 실행한 성공/실패 결과를
이 설정의 성능 결과로 해석하지 마세요.
