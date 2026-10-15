import math

import torch

from fold_metrics import keypoint_distance, normalized_score


def test_keypoint_distance():
    pairs = [(0, 1), (3, 2)]

    # 한 변이 0.2 m인 펼쳐진 정사각형
    flat_corners = torch.tensor([
        [0.0, 0.0, 0.0],  # 모서리 0
        [0.2, 0.0, 0.0],  # 모서리 1
        [0.2, 0.2, 0.0],  # 모서리 2
        [0.0, 0.2, 0.0],  # 모서리 3
    ])

    distance = keypoint_distance(flat_corners, pairs)
    assert math.isclose(distance, 0.2, abs_tol=1e-6)
    print("통과: 펼쳐진 천의 거리 = 0.2 m")

    # 모서리 0과 1, 모서리 3과 2가 각각 겹친 상태
    folded_corners = torch.tensor([
        [0.2, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.2, 0.2, 0.0],
        [0.2, 0.2, 0.0],
    ])

    distance = keypoint_distance(folded_corners, pairs)
    assert distance == 0.0
    print("통과: 완벽하게 접힌 천의 거리 = 0 m")


def test_normalized_score():
    score = normalized_score(p0=0.2, pt=0.1, p_opt=0.0)
    assert math.isclose(score, 0.5, abs_tol=1e-6)
    print("통과: 작을수록 좋은 지표의 점수 = 0.5")

    score = normalized_score(
        p0=0.0,
        pt=0.5,
        p_opt=1.0,
        lower_is_better=False,
    )
    assert math.isclose(score, 0.5, abs_tol=1e-6)
    print("통과: 클수록 좋은 지표의 점수 = 0.5")

    for lower_is_better in (True, False):
        try:
            normalized_score(
                p0=0.0,
                pt=0.5,
                p_opt=0.0,
                lower_is_better=lower_is_better,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("p0 == p_opt이면 ValueError가 필요합니다.")

    print("통과: p0 == p_opt일 때 ValueError 발생")


if __name__ == "__main__":
    test_keypoint_distance()
    test_normalized_score()
    print("모든 테스트 통과!")