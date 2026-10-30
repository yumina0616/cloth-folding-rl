# cloth-folding-rl

강화학습 기반 로봇팔 천 접기 프로젝트 (로봇비전캡스톤디자인).

- 환경: Isaac Sim / Isaac Lab, Franka Panda
- 상태 표현: 천 모서리 4개 keypoint
- 학습 파이프라인: 스크립트 기반 전문가 시연 생성 → BC 사전학습 → SAC 정책 개선 → 실패 보정(DAgger) → 통합 평가

## 협업 규칙

브랜치, 커밋, PR, 리뷰 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md), 실험 기록은 [experiments/](experiments/README.md)를 본다.

## 커밋하지 않는 것

학습 로그, 체크포인트(`*.pt` 등), 영상, Isaac Sim 캐시는 `.gitignore`로 제외한다.

## 가상 파지 모듈 (A)

단일 정점 가상 파지의 공통 인터페이스와 실행 방법은 [가상 파지 사용법](scripts/grasp/README_virtual_picker.md)을 참고한다. 기존 004 패치 파지 스크립트는 별도로 보존한다.

현재 가상 파지 접기 시연은 [직사각형 접기 실행 방법](scripts/grasp/README_rectangular_fold.md)을 따른다.
천 모서리를 집게에 가상으로 연결하고 두 모서리를 순서대로 옮긴 뒤, 두 쌍의 거리가 모두 2cm 이내인지 판정한다.
[파지 안정성 비교](scripts/grasp/README_grasp_stability.md)와 성공한 실행의 영상·시연 기록 저장을 포함한다.
학습된 정책이나 접촉 마찰 기반 물리 파지를 구현한 결과는 아니다.
