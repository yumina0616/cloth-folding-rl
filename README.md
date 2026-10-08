# cloth-folding-rl

강화학습 기반 로봇팔 천 접기 프로젝트 (로봇비전캡스톤디자인).

- 환경: Isaac Sim / Isaac Lab, Franka Panda
- 상태 표현: 천 모서리 4개 keypoint
- 학습 파이프라인: 스크립트 기반 전문가 시연 생성 → BC 사전학습 → SAC 정책 개선 → 실패 보정(DAgger) → 통합 평가

## 협업 규칙

브랜치, 커밋, PR, 리뷰 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md), 실험 기록은 [experiments/](experiments/README.md)를 본다.

## 커밋하지 않는 것

학습 로그, 체크포인트(`*.pt` 등), 영상, Isaac Sim 캐시는 `.gitignore`로 제외한다.
