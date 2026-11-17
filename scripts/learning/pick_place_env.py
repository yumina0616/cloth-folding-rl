"""Gymnasium macro environment; the backend executes real simulator actions."""
import numpy as np

from scripts.learning.policy_contract import PolicySpaces, encode_observation, decode_action
from scripts.learning.reward import compute_reward, check_fold_success


class PickPlaceEnv(PolicySpaces):
    def __init__(self, backend, grasp_mode, max_attempts=8, success_threshold=.02):
        if max_attempts < 1 or success_threshold <= 0:
            raise ValueError("max_attempts and success_threshold must be positive")
        self.backend = backend
        self.grasp_mode = grasp_mode
        self.max_attempts = max_attempts
        self.threshold = success_threshold
        self.finished = True

    def observation(self):
        return encode_observation(self.backend.corners().detach().cpu().numpy(),
                                  self.backend.pairs, self.grasp_mode)

    def reset(self, *, seed=None, options=None):
        # Call gym.Env rather than PolicySpaces' deliberately unavailable reset.
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        self.backend.reset(seed=seed)
        self.attempts = 0
        self.finished = False
        return self.observation(), {"grasp_mode": self.grasp_mode}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before stepping a completed episode")
        world_action = decode_action(action)
        before = self.backend.corners().clone()
        outcome = self.backend.execute(world_action)
        after = self.backend.corners().clone()
        if not np.isfinite(after.detach().cpu().numpy()).all():
            raise RuntimeError("Simulator returned non-finite cloth coordinates")
        self.attempts += 1
        success = check_fold_success(after, self.backend.pairs, self.threshold)
        dropped = bool(outcome.get("dropped", False))
        reward = float(compute_reward(before, after, self.backend.pairs, success=success, dropped=dropped))
        if not outcome.get("grasp_success", False):
            reward -= 1.
        terminated = success or dropped
        truncated = self.attempts >= self.max_attempts and not terminated
        self.finished = terminated or truncated
        distances = [float((after[i] - after[j]).norm()) for i, j in self.backend.pairs]
        info = {**outcome, "is_success": success, "grasp_mode": self.grasp_mode,
                "pair_distances_m": distances, "world_action": world_action.tolist(),
                "attempts": self.attempts}
        return self.observation(), reward, terminated, truncated, info

    def close(self):
        self.backend.close()
