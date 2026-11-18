"""Console height readout for keyboard teleoperation (no Isaac Sim; tested in tests/test_teleop_height_log.py).

All heights are above the table top, in cm:
  TCP     the fingertip centre (the point the keys move)
  cloth   the highest cloth node (shows whether anything was lifted at all)
  held    the cloth node being held, and how far it rose since the fingers closed
and, when given, the x-y distance from the TCP to the next pinch point in mm.
"""
from __future__ import annotations


def height_line(table_z, tcp_z, cloth_max_z, held_z=None, held_start_z=None, pinch_xy=None, guide="pinch"):
    """One readable line, e.g. '[height] TCP 8.3 cm | cloth max 0.4 cm | held 2.1 cm (+1.8 cm since close)'."""
    def cm(z):
        return (z - table_z) * 100

    line = f"[height] TCP {cm(tcp_z):5.1f} cm | cloth max {cm(cloth_max_z):5.1f} cm"
    if held_z is not None:
        line += f" | held {cm(held_z):5.1f} cm"
        if held_start_z is not None:
            line += f" ({(held_z - held_start_z) * 100:+.1f} cm since close)"
    if pinch_xy is not None:
        line += f" | {guide} xy {pinch_xy * 1000:4.0f} mm"
    return line


class HeightLogger:
    """Prints at most once per `interval_s` of simulation time, and only when a height changed.

    Changes smaller than `min_change_m` are ignored so a still arm does not flood the console.
    interval_s <= 0 turns the readout off.
    """

    def __init__(self, interval_s=0.5, min_change_m=0.001, printer=print):
        self.interval_s = interval_s
        self.min_change_m = min_change_m
        self.printer = printer
        self.reset()

    def reset(self):
        self._last_time = None
        self._last_values = None

    def update(self, time_s, table_z, tcp_z, cloth_max_z, held_z=None, held_start_z=None, pinch_xy=None,
               guide="pinch"):
        """Return the printed line, or None when nothing was printed."""
        if self.interval_s <= 0:
            return None
        if self._last_time is not None and time_s - self._last_time < self.interval_s:
            return None
        values = (tcp_z, cloth_max_z, held_z, pinch_xy)
        if self._last_values is not None and not self._changed(values):
            return None
        line = height_line(table_z, tcp_z, cloth_max_z, held_z, held_start_z, pinch_xy, guide)
        self.printer(line, flush=True)
        self._last_time, self._last_values = time_s, values
        return line

    def _changed(self, values):
        for new, old in zip(values, self._last_values):
            if (new is None) != (old is None):
                return True
            if new is not None and abs(new - old) >= self.min_change_m:
                return True
        return False
