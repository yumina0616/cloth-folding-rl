"""Height readout checks without Isaac Sim: python -m pytest tests/test_teleop_height_log.py"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/grasp"))
from teleop_height_log import HeightLogger, height_line  # noqa: E402


def make_logger(**kwargs):
    lines = []
    logger = HeightLogger(printer=lambda line, flush=True: lines.append(line), **kwargs)
    return logger, lines


def test_line_is_in_cm_above_the_table():
    line = height_line(table_z=0.5, tcp_z=0.583, cloth_max_z=0.504)
    assert line == "[height] TCP   8.3 cm | cloth max   0.4 cm"


def test_line_shows_held_node_and_rise_since_close():
    line = height_line(0.0, tcp_z=0.06, cloth_max_z=0.022, held_z=0.021, held_start_z=0.003)
    assert line.endswith("| held   2.1 cm (+1.8 cm since close)")


def test_prints_at_most_once_per_interval():
    logger, lines = make_logger(interval_s=0.5)
    for step in range(10):  # 0.0 .. 0.9 s, height changing every step
        logger.update(step * 0.1, 0.0, 0.10 + step * 0.01, 0.0)
    assert len(lines) == 2   # at 0.0 s and 0.5 s


def test_still_arm_does_not_flood_the_console():
    logger, lines = make_logger(interval_s=0.5)
    for step in range(20):
        logger.update(step * 0.5, 0.0, 0.10, 0.0015)
    assert len(lines) == 1


def test_grasp_start_and_end_are_printed_even_without_motion():
    logger, lines = make_logger(interval_s=0.5)
    logger.update(0.0, 0.0, 0.10, 0.0)
    logger.update(0.5, 0.0, 0.10, 0.0, held_z=0.0, held_start_z=0.0)   # fingers closed
    logger.update(1.0, 0.0, 0.10, 0.0)                                  # released
    assert len(lines) == 3


def test_zero_interval_turns_it_off_and_reset_prints_again():
    logger, lines = make_logger(interval_s=0)
    assert logger.update(0.0, 0.0, 0.1, 0.0) is None and not lines
    logger, lines = make_logger(interval_s=0.5)
    logger.update(0.0, 0.0, 0.1, 0.0)
    logger.reset()                       # new episode: time restarts at 0
    logger.update(0.0, 0.0, 0.1, 0.0)
    assert len(lines) == 2
