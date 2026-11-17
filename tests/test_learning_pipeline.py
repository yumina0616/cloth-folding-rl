"""Data semantics, episode splitting, PPO checkpoint transfer, and macro failures."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.learning.demonstration_dataset import load_episode, load_episodes, split_episodes
from scripts.learning.pick_place_env import PickPlaceEnv
from scripts.learning.policy_contract import encode_observation, normalize_action, decode_action, VERSION
from scripts.learning.train_bc import train

PAIRS = [(0, 3), (1, 2)]
CORNERS = np.array([[.35, -.2, .004], [.35, .1, .004], [.65, .1, .004], [.65, -.2, .004]], dtype=np.float32)


def episode(path, mode):
    path.mkdir(parents=True)
    (path / "context.json").write_text(json.dumps({"grasp_mode": mode, "pairs": PAIRS,
        "pick_xy": "measured TCP at manual close" if mode == "physical" else "measured cloth corner at grasp",
        "place_xy": "measured TCP immediately before opening"}), encoding="utf-8")
    (path / "metadata.json").write_text('{"success": true}', encoding="utf-8")
    (path / "high_level.jsonl").write_text(json.dumps({"observation": {"corners": CORNERS.tolist()},
        "action": {"pick": [.35, -.2], "place": [.65, -.2]}, "low_level_start": 0,
        "low_level_end": 2}) + "\n", encoding="utf-8")
    np.savez(path / "low_level.npz", tcp_w=np.array([[.36, -.19, .02], [.65, -.2, .01]]))
    return load_episode(path)


class FakeBackend:
    pairs = PAIRS

    def reset(self, seed=None):
        self.state = torch.tensor(CORNERS)

    def corners(self):
        return self.state

    def execute(self, action):
        self.action = action
        return {"grasp_success": False, "dropped": False}

    def close(self):
        pass


class LearningPipelineTest(unittest.TestCase):
    def test_contract_roundtrip_and_mode(self):
        action = [.35, -.2, .65, .1]
        np.testing.assert_allclose(decode_action(normalize_action(action)), action, atol=1e-7)
        virtual = encode_observation(CORNERS, PAIRS, "virtual")
        physical = encode_observation(CORNERS, PAIRS, "physical")
        self.assertEqual(virtual.shape, (30,))
        np.testing.assert_array_equal(virtual[-2:], [1, 0])
        np.testing.assert_array_equal(physical[-2:], [0, 1])
        with self.assertRaises(ValueError):
            normalize_action([99, 0, .5, 0])

    def test_virtual_pick_conversion_and_strict_missing_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            virtual = episode(root / "virtual", "virtual")
            physical = episode(root / "physical", "physical")
            np.testing.assert_allclose(decode_action(virtual.actions[0])[:2], [.36, -.19], atol=1e-7)
            np.testing.assert_allclose(decode_action(physical.actions[0])[:2], [.35, -.2], atol=1e-7)
            (root / "virtual/context.json").unlink()
            accepted, rejected = load_episodes([root])
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(rejected), 1)

    def test_split_never_leaks_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            episodes = [episode(Path(temporary) / str(i), mode) for i, mode in
                        enumerate(["virtual", "virtual", "physical", "physical"])]
            training, validation = split_episodes(episodes)
            self.assertFalse({e.path for e in training} & {e.path for e in validation})
            self.assertEqual({e.mode for e in training}, {"virtual", "physical"})
            self.assertEqual({e.mode for e in validation}, {"virtual", "physical"})

    def test_missed_grasp_is_negative_and_horizon_is_truncation(self):
        backend = FakeBackend()
        env = PickPlaceEnv(backend, "physical", max_attempts=1)
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertLess(reward, -1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["is_success"])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4))

    def test_drop_terminates_and_is_penalized(self):
        backend = FakeBackend()
        backend.execute = lambda action: {"grasp_success": True, "dropped": True}
        env = PickPlaceEnv(backend, "physical", max_attempts=1)
        env.reset()
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["dropped"])
        self.assertLess(reward, -5)

    def test_success_requires_both_pairs(self):
        backend = FakeBackend()
        def fold(action):
            for source, destination in PAIRS:
                backend.state[source] = backend.state[destination]
            return {"grasp_success": True, "dropped": False}
        backend.execute = fold
        env = PickPlaceEnv(backend, "virtual")
        env.reset()
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertGreater(reward, 10)

    def test_bc_checkpoint_runs_and_continues_with_ppo(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = [episode(root / str(i), mode) for i, mode in
                        enumerate(["virtual", "virtual", "physical", "physical"])]
            report = train(episodes, root / "bc", epochs=2)
            model = PPO.load(root / "bc", device="cpu")
            self.assertEqual(model.cloth_contract, VERSION)
            self.assertEqual(model.cloth_training_modes, ["physical", "virtual"])
            self.assertTrue(np.isfinite(report["best_validation_coordinate_rmse_m"]))
            env = PickPlaceEnv(FakeBackend(), "physical", max_attempts=1)
            model.set_env(env)
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            self.assertEqual(action.shape, (4,))
            model.learn(total_timesteps=32)
            self.assertEqual(model.num_timesteps, 32)


if __name__ == "__main__":
    unittest.main()
