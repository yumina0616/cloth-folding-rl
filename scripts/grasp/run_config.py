"""Shared run settings from a JSON file in configs/ (no Isaac Sim needed).

Keys starting with "_" are notes and are skipped. Every other key must be an option of the
script that loads it; the file only changes defaults, so a command-line option still wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "cloth_unified.json"


def load_config(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the config must be a JSON object")
    return {key: value for key, value in data.items() if not key.startswith("_")}


def apply_config_defaults(parser: argparse.ArgumentParser, argv) -> Path | None:
    """Read --config from argv (default DEFAULT_CONFIG, 'none' = skip) and use it as option defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    if known.config.lower() == "none":
        return None
    path = Path(known.config)
    values = load_config(path)
    options = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - options)
    if unknown:
        parser.error(f"{path}: unknown settings {unknown}")
    parser.set_defaults(**values)
    return path
