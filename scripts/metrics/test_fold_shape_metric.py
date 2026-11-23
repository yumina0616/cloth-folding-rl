import numpy as np
import pytest
from fold_shape_metric import top_view_iou, fold_shape_success


INITIAL = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0.3, 0.3, 0]], float)
PAIRS = [(0, 1), (2, 3)]


def grid(x0, x1, y0, y1, n=40):
    xx, yy = np.meshgrid(np.linspace(x0,x1,n), np.linspace(y0,y1,n))
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n*n)))


def folded_corners(dx=0):
    return np.array([[.3+dx,0,0],[.3+dx,0,0],[.3+dx,.3,0],[.3+dx,.3,0]])


def score(nodes, corners=None, **kwargs):
    return top_view_iou(nodes, INITIAL, folded_corners() if corners is None else corners,
                        PAIRS, cloth_size=.3, mesh_spacing=.3/39, pixel_size=.002, **kwargs)


def test_neat_fold_high_iou():
    m = score(grid(.15,.3,0,.3))
    assert m['iou'] > .85
    assert m['inclusion'] > .85


def test_unfolded_sheet_low_iou():
    assert score(grid(0,.3,0,.3))['iou'] < .65


def test_crumpled_pile_low_iou_but_high_inclusion():
    m = score(grid(.23,.28,.11,.19))
    assert m['inclusion'] > .90
    assert m['iou'] < .25


def test_sliding_destination_relative():
    dx = .08
    slid = grid(.15+dx,.3+dx,0,.3)
    relative = score(slid, folded_corners(dx=dx))['iou']
    fixed = score(slid, folded_corners(dx=dx), relative_to_destination=False)['iou']
    assert relative > .85
    assert fixed < .5


def test_fold_shape_success_threshold():
    ok, m = fold_shape_success(grid(.15,.3,0,.3), INITIAL, folded_corners(), PAIRS,
                               cloth_size=.3, mesh_spacing=.3/39, pixel_size=.002, min_iou=.75)
    assert ok and m['iou'] > .75


def test_invalid_values():
    with pytest.raises(ValueError):
        top_view_iou(grid(.15,.3,0,.3), INITIAL, folded_corners(), PAIRS, pixel_size=-.1)
