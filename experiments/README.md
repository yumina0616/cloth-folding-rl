# 실험 목록

새 실험: `_template.md`를 `NNN-<slug>.md`로 복사해서 쓴다. 규칙은 [CONTRIBUTING.md](../CONTRIBUTING.md) 참고.

- 이긴 실험이든 진 실험이든 노트는 `main`에 넣는다 (실패 기록 = 보고서 근거).
- 영상과 로그: Google Drive 공유 폴더 `<run_id>/` <!-- TODO: 폴더 링크 -->
- run_id 형식: `YYYYMMDD-HHMM_<브랜치>_s<seed>` (예: `20261008-1530_grasp-table-edge_s0`)

| # | 문제 | 아이디어 | grasp | 결과 | 판정 | PR |
|---|---|---|---|---|---|---|
| [001](001-grasp-direct-pinch.md) | 물리 파지 | 직접 하강 후 집기 | physical | 천이 손가락 사이에 안 들어옴 | 실패 | - |
| [002](002-grasp-tilted-approach.md) | 물리 파지 | 기울어진 자세 접근 | physical | 접촉 O, 상승 X | 실패 | - |
| [003](003-grasp-high-friction.md) | 물리 파지 | 마찰 증가 (손가락 5.0) | physical | 천이 따라오지 않음 | 실패 | - |
| [004](004-grasp-virtual-picker.md) | 물리 파지 우회 | 정점 4개 연결 | virtual | 약 46.9 mm 상승 | 성공 | - |
| [005](005-grasp-table-edge.md) | 물리 파지 | 테이블 밖으로 밀고 측면 집기 | physical | 돌출 38.5~41.5 mm, 상승 X | 부분 성공 | - |
