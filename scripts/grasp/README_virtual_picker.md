# A: 단일 정점 가상 파지

> 이 문서는 초기 강제 보정 방식의 인터페이스와 비교용 예제를 설명합니다.
> 현재 기본 시연은 [PhysX 연결 방식의 안정성 진단](README_grasp_stability.md)과
> [직사각형 접기](README_rectangular_fold.md)를 사용합니다.

`virtual_picker.py`는 천 정점 하나를 선택해 그리퍼 위치를 따라가게 하는 공통 모듈입니다.
004의 여러 정점 패치 방식은 기존 `grasp_cloth_corner.py`에 보존하며,
사용자 결정에 따라 A의 새 인터페이스에는 단일 정점 방식을 사용합니다.

## 실행 예제

```powershell
cd C:\IsaacLab
.\isaaclab.bat -p "C:\cloth-folding-rl\scripts\grasp\virtual_grasp_single_point.py" --grasp_mode virtual --viz kit presets=isaacsim_physx
```

접근 → 하강 → 집게 닫기 → 가상 집기 → 8 cm 들기 → 10초 정지 관찰 →
내리기 → 가상 고정 해제 → 집게 열기 → 후퇴합니다.
`--move_distance 0.04`를 추가하면 관찰 후 옆으로 이동합니다.
예제의 굽힘 강성은 사용자가 육안상 요동 감소를 확인한 2e6이며,
`--bend_stiffness`로 변경할 수 있습니다. 이 설정은 공통 모듈에 들어 있지 않습니다.

기본 측정 파일은 `runs/virtual_single_point_observation.csv`입니다.
재실행 시 덮어쓰므로 비교 실행은 `--metrics_path`로 다른 경로를 지정합니다.
PASS는 집게 움직임·상승·해제 검사이며, 천 전체 진동의 안정성 판정은 아닙니다.

## 다른 스크립트에서 사용

같은 `scripts/grasp` 폴더의 스크립트에서는 다음처럼 import합니다.
저장소 루트에서 실행하는 모듈에서는 `scripts.grasp.virtual_picker`로 import할 수 있습니다.
로봇 이동과 집게 열기·닫기 명령은 호출하는 코드가 담당합니다.

```python
from virtual_picker import as_tensor, create_picker

base = env.unwrapped
cloth = base.scene["deformable"]
ee = base.scene["ee_frame"]
picker = create_picker(
    cloth,
    lambda: as_tensor(ee.data.target_pos_w)[0, 0],
    grasp_mode="virtual",
    grasp_radius=0.03,
)

# 먼저 로봇을 천 근처로 이동하고 집게를 닫습니다.
# position_w: world 좌표(m), shape (3,).
if not picker.grasp(position_w):
    raise RuntimeError("집기 위치 또는 그리퍼가 선택 정점에서 너무 멉니다")

# cfg.decimation=1일 때의 매 스텝 호출 예시.
if picker.backend == "kinematic":
    picker.update(dt=base.physics_dt)
env.step(action)
if picker.backend == "state_projection":
    picker.update(dt=base.physics_dt)

picker.release()  # 이후 로봇 집게 열기
picker.reset()    # 반드시 env.reset() 전에 호출
env.reset()
```

| 인터페이스 | 설명 |
|---|---|
| `grasp(position_w, vertex_id=None)` | 가장 가까운 정점 1개를 잡기. 성공 True / 반경 밖 False |
| `update(dt)` | 그리퍼를 따라 파지 목표 갱신 |
| `release()` | 연결 해제. 반복 호출 가능 |
| `reset()` | 연결과 내부 상태 정리 |
| `is_grasping`, `vertex_id` | 파지 여부와 선택 정점 번호 |
| `target_position_w` | world 좌표의 파지 목표, 해제 상태면 None |

`grasp_radius`는 정점 선택과 그리퍼 도달 판정 거리이며 여러 정점을 잡는 반경이 아닙니다.
그리퍼 기준점은 callback으로 전달하므로 필요하면 손가락 중심 중간점으로 교체할 수 있습니다.
초기 world 위치 오프셋을 유지하고 회전은 추종하지 않습니다.

표면 천은 물리 계산 직후 선택 정점의 위치·속도를 덮어쓰는 근사 방식입니다.
볼륨 변형체에서 kinematic target이 제공되면 물리 계산 전에 고정 목표를 갱신합니다.
decimation > 1이면 환경 내부의 각 물리 스텝에 해당 호출을 연결해야 합니다.
한 picker는 환경 하나와 정점 하나를 담당하며 같은 정점을 여러 picker가 제어하면 안 됩니다.
학습 루프의 자동 환경 초기화에도 reset을 연결해야 합니다.

## 모듈 검사

Isaac Sim을 시작하지 않고 Torch가 있는 Python에서 검사할 수 있습니다.

```powershell
& C:\IsaacLab\_isaac_sim\kit\python\python.exe C:\cloth-folding-rl\tests\test_virtual_picker.py
```

2026-10-15 검증: 저장소 전체 테스트 10개(기존 모서리 추적 4개, 새 파지 모듈 6개)가
통과했습니다. 이 저장소 경로에서 PhysX 예제를 화면 없이 실행해 집게 닫기·열기,
8 cm 들기, 10초 관찰, 해제·후퇴 검사도 통과했습니다. 천 전체의 육안상 안정성은
이 PASS 판정에 포함하지 않습니다.
