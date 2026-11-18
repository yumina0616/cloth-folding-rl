"""Where to grasp next, where to put it down, and whether the hand is there (no Isaac Sim).

Tested in tests/test_teleop_pinch_guide.py. The teleop script draws one sphere:
  fingers open    on the next pinch point (grasp here); compared with the TCP
  fingers closed  on the destination's pinch point (put the held point here); compared with the
                  held cloth point, so slip in the fingers is included
Red while farther than the tolerance in x-y, green within it. Height is not part of the check.

Why the destination's pinch point: pinch points sit inward from their corner toward the cloth
centre by the same amounts, so folding one corner onto another also lays the source pinch point
onto the destination pinch point (the demo's --place_mode mirror target).
"""
from __future__ import annotations

import torch

RED, GREEN = 0, 1          # marker prototype indices
DONE_FRACTION = 0.5        # a pair counts as folded once its corner gap shrank below this share of the start gap


def next_source_slot(pairs, initial_corners_xy, corners_xy, done_fraction=DONE_FRACTION):
    """Corner slot to grasp next, or None when every pair is folded.

    pairs: [(source, destination), ...] in folding order (rectangular_pairs).
    A pair is done when its x-y corner gap fell below done_fraction of the gap at reset.
    Not counting K presses: a failed grasp (closed and opened without lifting) keeps the same point.
    Not using how far the source moved: the second source corner can be dragged while the first
    pair is folded, and that must not skip it.
    """
    for source, destination in pairs:
        start = torch.linalg.vector_norm(initial_corners_xy[source] - initial_corners_xy[destination])
        now = torch.linalg.vector_norm(corners_xy[source] - corners_xy[destination])
        if float(now) >= done_fraction * float(start):
            return source
    return None


def pair_with_source(pairs, source):
    """The (source, destination) pair that starts at this corner slot, or None."""
    return next(((s, d) for s, d in pairs if s == source), None)


def xy_distance(tcp, point):
    """Horizontal distance between the TCP and a point, in m."""
    return float(torch.linalg.vector_norm(tcp[:2] - point[:2]))


def marker_color(distance, tolerance):
    return GREEN if distance <= tolerance else RED
