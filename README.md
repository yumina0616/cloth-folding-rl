# cloth-folding-rl

강화학습 기반 로봇팔 천 접기 프로젝트 (로봇비전캡스톤디자인).

- 환경: Isaac Sim / Isaac Lab, Franka Panda
- 상태 표현: 천 모서리 4개 keypoint
- 학습 파이프라인: 스크립트 기반 전문가 시연 생성 → BC 사전학습 → SAC 정책 개선 → 실패 보정(DAgger) → 통합 평가

## 브랜치 규칙

- `main`: 동작이 확인된 코드만 둔다. 직접 push하지 않는다.
- 작업은 `feature/<기능명>` 브랜치에서 하고 Pull Request로 머지한다.
  - 예: `feature/corner-tracker`, `feature/virtual-picker`
- PR 설명에 실험 결과(실패한 시도 포함)를 함께 적는다.

## 커밋하지 않는 것

학습 로그, 체크포인트(`*.pt` 등), 영상, Isaac Sim 캐시는 `.gitignore`로 제외한다.
