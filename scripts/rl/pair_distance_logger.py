"""Record the distance of every target corner pair, per step and per environment, to a CSV file.

The reward uses the mean over pairs, so this file is where each pair can be checked later
(e.g. "in env 3, pair 0-1 closed at step 40 but pair 2-3 never did").

One row per (step, environment):

    step,env_id,pair_0_1,pair_2_3
    0,0,0.2000,0.2000
    0,1,0.1985,0.2011
    ...

Distances are in m. Open with Excel or pandas.read_csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.rl.fold_mdp import pair_distances


class PairDistanceLogger:
    def __init__(self, filepath, pairs):
        self.filepath = Path(filepath)
        self.pairs = list(pairs)
        self.columns = ["step", "env_id"] + [f"pair_{i}_{j}" for i, j in self.pairs]

        # 새 파일로 시작하고 헤더만 먼저 쓴다
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.columns)

    def record(self, step, corners):
        """corners (N, 4, 3) -> append N rows. Written immediately so a crash keeps earlier steps."""
        distances = pair_distances(corners, self.pairs).detach().cpu().tolist()  # (N, P)
        with self.filepath.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for env_id, row in enumerate(distances):
                writer.writerow([int(step), env_id] + [f"{d:.4f}" for d in row])
