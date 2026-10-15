import torch


def keypoint_distance(
    corners: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> float:
    """겹쳐야 하는 모서리 쌍의 평균 3D 거리(m)를 반환한다.

    corners: 모서리 4개의 좌표. shape (4, 3), 단위 m.
    pairs: 서로 겹쳐야 하는 모서리 번호 쌍.
    """
    if corners.shape != (4, 3):
        raise ValueError("corners의 shape은 (4, 3)이어야 합니다.")

    if not pairs:
        raise ValueError("pairs에는 모서리 쌍이 하나 이상 필요합니다.")

    distances = []

    for i, j in pairs:
        if not (0 <= i < 4 and 0 <= j < 4):
            raise ValueError("모서리 번호는 0~3이어야 합니다.")

        distance = torch.linalg.vector_norm(corners[i] - corners[j])
        distances.append(distance)

    return torch.stack(distances).mean().item()


def normalized_score(
    p0: float,
    pt: float,
    p_opt: float,
    lower_is_better: bool = True,
) -> float:
    """시작 상태에서 최적 상태까지의 개선 정도를 계산한다."""
    if p0 == p_opt:
        raise ValueError("p0와 p_opt가 같으면 점수를 계산할 수 없습니다.")

    if lower_is_better:
        return (p0 - pt) / (p0 - p_opt)

    return (pt - p0) / (p_opt - p0)