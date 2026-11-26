"""PPO policy update for two separate BC-pretrained SB3 PPO actor-critics.

This module only *updates* collected on-policy episodes; it does not launch Isaac,
reset its cloth, or collect trajectories. Call `sample_action` at each policy
*decision*, append PPOTransition after the outcome, and `update_two_policies`
only after complete trajectories have been collected. Do not reuse old rollouts.

High-policy reward should credit an entire scripted-pick + low-level option,
while low-policy reward is the bridge's per-active-action reward. Never add
SETTLE frames as low-policy decisions. Each `PPOTransition` sequence must end
with done=True (no terminal bootstrap). Truncation bootstrapping is NOT handled.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PPOTransition:
    observation: np.ndarray
    action: np.ndarray  # RAW policy sample; not decoded/clipped IK action
    log_prob: float
    value: float
    reward: float
    done: bool


def sample_action(model, observation):
    """Sample a PPO action and save its original log-probability/value."""
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != model.observation_space.shape or not np.isfinite(obs).all():
        raise ValueError(f"Expected finite observation {model.observation_space.shape}")
    with torch.no_grad():
        tensor = torch.as_tensor(obs[None], device=model.device)
        actions, values, log_probs = model.policy(tensor, deterministic=False)
    return (actions[0].detach().cpu().numpy().astype(np.float32),
            float(values[0].item()), float(log_probs[0].item()))


def advantages_from_complete_episodes(episodes, *, gamma=0.99, gae_lambda=0.95):
    """GAE for COMPLETE trajectories only; concatenated episodes don't leak returns."""
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    if not episodes or any(not ep or ep[-1].done is not True for ep in episodes):
        raise ValueError("Only nonempty, fully terminated trajectories are supported")
    flat, advantages, returns = [], [], []
    for episode in episodes:
        ep_adv = np.zeros(len(episode), dtype=np.float32)
        next_value, next_advantage = 0.0, 0.0
        for i in range(len(episode) - 1, -1, -1):
            t = episode[i]
            if not all(np.isfinite([t.reward, t.value, t.log_prob])):
                raise ValueError("Nonfinite rollout data")
            if t.done and i != len(episode) - 1:
                raise ValueError("Split trajectories at episode boundaries")
            nonterminal = 0.0 if t.done else 1.0
            delta = t.reward + gamma * next_value * nonterminal - t.value
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            ep_adv[i] = next_advantage
            next_value = t.value
        flat.extend(episode)
        advantages.extend(ep_adv.tolist())
        returns.extend((ep_adv + np.asarray([t.value for t in episode])).tolist())
    return flat, np.asarray(advantages, dtype=np.float32), np.asarray(returns, dtype=np.float32)


@torch.enable_grad()
def ppo_update(model, episodes, *, epochs=4, minibatch_size=64,
               gamma=0.99, gae_lambda=0.95, clip_range=0.2,
               value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5):
    """One clipped-PPO update on fresh complete episodes for one SB3 PPO model."""
    if epochs < 1 or minibatch_size < 1 or clip_range <= 0 or max_grad_norm <= 0:
        raise ValueError("Invalid PPO hyperparameters")
    flat, advantages, returns = advantages_from_complete_episodes(
        episodes, gamma=gamma, gae_lambda=gae_lambda)
    obs = np.stack([np.asarray(t.observation, dtype=np.float32) for t in flat])
    act = np.stack([np.asarray(t.action, dtype=np.float32) for t in flat])
    if obs.shape[1:] != model.observation_space.shape or act.shape[1:] != model.action_space.shape:
        raise ValueError("Rollout observation/action shapes differ from policy spaces")
    if not np.isfinite(obs).all() or not np.isfinite(act).all():
        raise ValueError("Nonfinite observations/actions")
    device = model.device
    observations = torch.as_tensor(obs, device=device)
    actions = torch.as_tensor(act, device=device)
    old_log_probs = torch.as_tensor([t.log_prob for t in flat], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)
    if len(flat) > 1 and float(advantages.std(unbiased=False)) > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    model.policy.set_training_mode(True)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(flat), device=device).split(minibatch_size):
            values, log_probs, entropy = model.policy.evaluate_actions(
                observations[indices], actions[indices])
            values = values.flatten()
            ratio = torch.exp(log_probs - old_log_probs[indices])
            surrogate = ratio * advantages[indices]
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages[indices]
            policy_loss = -torch.minimum(surrogate, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[indices])
            entropy_loss = (-entropy.mean() if entropy is not None else log_probs.mean())
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.policy.set_training_mode(False)
    return {"samples": len(flat), "updates": len(losses), "mean_loss": float(np.mean(losses))}


def update_two_policies(high_model, low_model, high_episodes, low_episodes, **kwargs):
    """Update high and low independently. Both must contain new complete rollouts."""
    if not high_episodes or not low_episodes:
        raise ValueError("Both high and low trajectories are required")
    return {"high": ppo_update(high_model, high_episodes, **kwargs),
            "low": ppo_update(low_model, low_episodes, **kwargs)}
