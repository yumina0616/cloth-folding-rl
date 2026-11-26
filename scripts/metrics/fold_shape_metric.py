"""Top-view cloth silhouette metric for a half-fold (Issue #39).

NumPy only. The projected silhouette is approximated by covering each cloth mesh
node with a small disk. This is a resolution-dependent approximation, not an
exact triangle-mesh projection. Calibrate threshold on saved successful/failing
runs before using it to filter demonstration data.
"""
from __future__ import annotations

import numpy as np


def _xy(values, name, shape_end):
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (2, 3) or (shape_end is not None and values.shape[0] != shape_end):
        raise ValueError(f'{name} must have shape ({shape_end if shape_end is not None else "N"}, 2 or 3)')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} contains nonfinite values')
    return values[:, :2]


def goal_half_rectangle(initial_corners, pairs):
    """Ideal folded half-sheet polygon from initial corners and directed pairs.

    pairs are (moving_corner, destination_corner); the target half is the half
    containing the two stationary destination corners. Retains original yaw.
    """
    corners = _xy(initial_corners, 'initial_corners', 4)
    if len(pairs) != 2 or len({int(i) for pair in pairs for i in pair}) != 4:
        raise ValueError('pairs must cover four distinct corners in two pairs')
    (s0, d0), (s1, d1) = pairs
    if any(type(i) not in (int, np.int32, np.int64) or not 0 <= i < 4 for i in (s0,d0,s1,d1)):
        raise ValueError('invalid corner index')
    poly = np.array([corners[d0], corners[d1],
                     0.5 * (corners[d1] + corners[s1]),
                     0.5 * (corners[d0] + corners[s0])])
    area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
                     np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    if area < 1e-8:
        raise ValueError('degenerate goal rectangle')
    return poly


def _polygon_mask(xx, yy, polygon):
    inside = np.zeros_like(xx, dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = a
        x2, y2 = b
        cross = (y1 > yy) != (y2 > yy)
        x_intersect = (x2 - x1) * (yy - y1) / (y2 - y1 if abs(y2-y1) > 1e-15 else 1e-15) + x1
        inside ^= cross & (xx < x_intersect)
    return inside


def top_view_iou(nodes_w, initial_corners, current_corners, pairs,
                 *, cloth_size=0.30, mesh_spacing=None, pixel_size=0.003,
                 relative_to_destination=True):
    """Compute silhouette IoU, inclusion, and coverage on a top-view grid.

    Current cloth is rasterized as disks centered on ALL mesh nodes (N, 3).
    Approximate disk radius is 0.75 times the rest-pose mesh spacing. Use
    mesh_spacing from actual flat mesh if the topology is not a square grid.

    The goal follows the current destination-edge midpoint by translation if
    relative_to_destination is True, avoiding arbitrary world-position penalty
    when the entire towel slides. Destination-edge rotation isn't corrected.
    """
    nodes = _xy(nodes_w, 'nodes_w', None)
    initial = _xy(initial_corners, 'initial_corners', 4)
    current = _xy(current_corners, 'current_corners', 4)
    if len(nodes) < 4:
        raise ValueError('at least four cloth mesh nodes required')
    if not (np.isfinite(cloth_size) and cloth_size > 0 and np.isfinite(pixel_size) and pixel_size > 0):
        raise ValueError('cloth_size and pixel_size must be finite and positive')
    if mesh_spacing is None:
        # Assumes the rest mesh has approximately sqrt(N) points per side.
        mesh_spacing = cloth_size / max(np.sqrt(len(nodes)) - 1, 1)
    if not (np.isfinite(mesh_spacing) and mesh_spacing > 0):
        raise ValueError('mesh_spacing must be finite and positive')
    radius = 0.75 * mesh_spacing
    goal = goal_half_rectangle(initial, pairs)
    if relative_to_destination:
        destination_ids = [pairs[0][1], pairs[1][1]]
        translation = current[destination_ids].mean(axis=0) - initial[destination_ids].mean(axis=0)
        goal = goal + translation

    low = np.minimum(nodes.min(axis=0) - radius, goal.min(axis=0)) - pixel_size
    high = np.maximum(nodes.max(axis=0) + radius, goal.max(axis=0)) + pixel_size
    dims = np.ceil((high - low) / pixel_size).astype(int) + 1
    if np.prod(dims) > 5_000_000:
        raise ValueError('raster grid too large; increase pixel_size')
    x = low[0] + (np.arange(dims[0]) + 0.5) * pixel_size
    y = low[1] + (np.arange(dims[1]) + 0.5) * pixel_size
    xx, yy = np.meshgrid(x, y, indexing='ij')
    target = _polygon_mask(xx, yy, goal)
    occupied = np.zeros_like(target)
    for nx, ny in nodes:
        ix0 = max(0, int(np.floor((nx - radius - low[0]) / pixel_size)))
        ix1 = min(dims[0], int(np.ceil((nx + radius - low[0]) / pixel_size)) + 1)
        iy0 = max(0, int(np.floor((ny - radius - low[1]) / pixel_size)))
        iy1 = min(dims[1], int(np.ceil((ny + radius - low[1]) / pixel_size)) + 1)
        occupied[ix0:ix1, iy0:iy1] |= ((xx[ix0:ix1, iy0:iy1]-nx)**2 +
                                       (yy[ix0:ix1, iy0:iy1]-ny)**2 <= radius**2)
    intersection = np.count_nonzero(occupied & target)
    union = np.count_nonzero(occupied | target)
    occupied_count = np.count_nonzero(occupied)
    target_count = np.count_nonzero(target)
    if union == 0 or occupied_count == 0 or target_count == 0:
        raise ValueError('empty projected silhouette or goal mask')
    return {
        'iou': float(intersection / union),
        'inclusion': float(intersection / occupied_count),
        'coverage': float(intersection / target_count),
        'projected_area_m2': float(occupied_count * pixel_size**2),
        'target_area_m2': float(target_count * pixel_size**2),
    }


def fold_shape_success(nodes_w, initial_corners, current_corners, pairs, *,
                       min_iou=0.75, **kwargs):
    if not 0 <= min_iou <= 1:
        raise ValueError('min_iou must be between 0 and 1')
    metrics = top_view_iou(nodes_w, initial_corners, current_corners, pairs, **kwargs)
    return metrics['iou'] >= min_iou, metrics
