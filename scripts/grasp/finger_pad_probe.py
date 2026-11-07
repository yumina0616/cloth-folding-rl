"""Measure the Franka finger pads in the running simulation.

The grasp point (physical_grasp_xy) needs the pad size, which was only an estimate (pad_half).
This reads the collision geometry of the two finger links from the USD stage and reports, in
world coordinates and millimetres:
  - the bounding box of each whole finger
  - the extent of the fingertip only (points within tip_height of the lowest point),
    which is the part that touches the cloth

Call probe_finger_pads() once the hand is open and above the cloth. It never raises: a failure
is returned as {"error": ...} so a demonstration run is not lost because of the measurement.
"""
from __future__ import annotations

import numpy as np

FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")


def tip_extents(points_w, tip_height=0.01):
    """Bounding box of all points and of the fingertip slice, sizes in metres.

    points_w: (N, 3) world points. The tip is the points within tip_height of the lowest z.
    """
    points = np.asarray(points_w, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_w must be a non-empty (N, 3) array")
    tip = points[points[:, 2] <= points[:, 2].min() + tip_height]
    return {
        "whole_min": points.min(axis=0), "whole_size": points.max(axis=0) - points.min(axis=0),
        "tip_min": tip.min(axis=0), "tip_size": tip.max(axis=0) - tip.min(axis=0),
        "tip_points": int(len(tip)),
    }


def _mm(values):
    return [round(float(v) * 1000, 1) for v in values]


def _finger_points(stage, link_prim):
    """World points of the collision geometry under one finger link (all geometry if none is tagged)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    geometry = [p for p in Usd.PrimRange(link_prim) if p.IsA(UsdGeom.Gprim)]
    collision = [p for p in geometry if p.HasAPI(UsdPhysics.CollisionAPI)]
    used = collision if collision else geometry
    points = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    for prim in used:
        if prim.IsA(UsdGeom.Mesh):
            local = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if local is None:
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            points.extend(tuple(matrix.Transform(p)) for p in local)
        else:
            # Primitive shapes (cube, cylinder, ...): use the corners of the world bounding box
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = box.GetMin(), box.GetMax()
            points.extend((x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))
    return np.array(points, dtype=float), [str(p.GetPath()) for p in used], bool(collision)


def probe_finger_pads(tip_height=0.01):
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        report = {}
        for name in FINGER_LINKS:
            prims = [p for p in stage.Traverse() if p.GetName() == name]
            if not prims:
                report[name] = {"error": "link not found on the stage"}
                continue
            points, paths, tagged = _finger_points(stage, prims[0])
            if len(points) == 0:
                report[name] = {"error": "no geometry points", "link": str(prims[0].GetPath())}
                continue
            ext = tip_extents(points, tip_height)
            report[name] = {
                "link": str(prims[0].GetPath()), "geometry_paths": paths[:6], "collision_tagged": tagged,
                "whole_size_mm": _mm(ext["whole_size"]), "tip_size_mm": _mm(ext["tip_size"]),
                "tip_center_xy_m": [round(float(v), 4) for v in (ext["tip_min"] + ext["tip_size"] / 2)[:2]],
                "tip_points": ext["tip_points"],
            }
        return report
    except Exception as exc:  # measurement only: never break the run
        return {"error": f"{type(exc).__name__}: {exc}"}
