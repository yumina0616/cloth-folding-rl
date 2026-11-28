"""Integration regression tests for correction episodes; run from repository root.

Usage: python -m unittest discover -s . -p test_loader_corrections.py
Requires numpy and the project's Python dependencies.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from scripts.learning.demonstration_dataset import load_episode


class CorrectionLoaderTest(unittest.TestCase):
    def make_episode(self, indices):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        (folder / 'context.json').write_text(json.dumps({
            'grasp_mode': 'physical',
            'pick_xy': 'measured TCP at manual close',
            'place_xy': 'measured TCP immediately before opening',
            'pairs': [[0, 1], [2, 3]],
        }))
        (folder / 'metadata.json').write_text(json.dumps({'success': True}))
        records = []
        n = len(indices) * 3
        corners = np.tile(np.array([
            [[.4, 0, 0], [.6, 0, 0], [.4, .2, 0], [.6, .2, 0]]
        ], dtype=np.float32), (n, 1, 1))
        for i, idx in enumerate(indices):
            r = {
                'low_level_start': i * 3,
                'low_level_end': i * 3 + 3,
                'observation': {'corners': corners[i * 3].tolist()},
                'action': {'pick': [.4, 0.]},
            }
            if idx is not None:
                r['pair_index'] = idx
            records.append(r)
        (folder / 'high_level.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        phases = np.array(['carry'] * n)
        actions = np.zeros((n, 8), dtype=np.float32)
        actions[:, 3] = 1.0
        np.savez(folder / 'low_level.npz',
                 time_s=np.arange(n, dtype=np.float32) * 0.1,
                 phase=phases,
                 action=actions,
                 corners_w=corners,
                 tcp_w=np.tile(np.array([[.4, 0., .1]], dtype=np.float32), (n, 1)),
                 grasping=np.ones(n, dtype=bool))
        return folder

    def test_correction_reuses_pair_zero(self):
        episode = load_episode(self.make_episode([0, 1, 0, 0]))
        self.assertEqual(episode.high_observations.shape, (4, 30))
        self.assertEqual(episode.low_observations.shape, (8, 28))
        for pick, index in enumerate([0, 1, 0, 0]):
            np.testing.assert_array_equal(episode.low_observations[pick * 2, 23:25],
                                          [1, 0] if index == 0 else [0, 1])

    def test_legacy_two_picks_without_pair_index(self):
        episode = load_episode(self.make_episode([None, None]))
        self.assertEqual(episode.high_observations.shape, (2, 30))
        np.testing.assert_array_equal(episode.low_observations[0, 23:25], [1, 0])
        np.testing.assert_array_equal(episode.low_observations[2, 23:25], [0, 1])

    def test_correction_missing_pair_index_fails(self):
        with self.assertRaisesRegex(ValueError, 'missing pair_index'):
            load_episode(self.make_episode([0, 1, None]))

    def test_invalid_pair_index_fails(self):
        with self.assertRaisesRegex(ValueError, 'pair_index'):
            load_episode(self.make_episode([0, 1, 2]))

    def test_physical_release_all_steps_included(self):
        folder = self.make_episode([0, 1])

        # 첫 번째 Pick/Place 구간을 carry + physical_release로 구성
        with np.load(folder / "low_level.npz") as data:
            log = {key: data[key].copy() for key in data.files}

        log["phase"] = np.array([
            "carry",
            "physical_release",
            "physical_release",
            "carry",
            "carry",
            "carry",
        ])
        log["action"][1:3, 7] = 1.0

        np.savez(folder / "low_level.npz", **log)

        episode = load_episode(folder)

        # 첫 번째 구간에서 열기 명령 2개가 모두 포함되어야 함
        np.testing.assert_array_equal(
            episode.low_actions[:2, 7],
            [1.0, 1.0],
        )

        self.assertEqual(len(episode.low_actions), 4)


if __name__ == '__main__':
    unittest.main()
