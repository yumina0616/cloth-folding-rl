import numpy as np
import pytest
import torch

from scripts.rl.ppo_policy_update import PPOTransition, advantages_from_complete_episodes, ppo_update


class FakePolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = torch.nn.Linear(obs_dim, act_dim)
        self.critic = torch.nn.Linear(obs_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def set_training_mode(self, mode):
        self.train(mode)

    def evaluate_actions(self, observations, actions):
        distribution = torch.distributions.Normal(self.actor(observations), torch.ones_like(actions))
        return (self.critic(observations), distribution.log_prob(actions).sum(-1),
                distribution.entropy().sum(-1))


class FakeModel:
    def __init__(self, obs_dim, act_dim):
        self.device = torch.device('cpu')
        self.observation_space = type('Space', (), {'shape': (obs_dim,)})()
        self.action_space = type('Space', (), {'shape': (act_dim,)})()
        self.policy = FakePolicy(obs_dim, act_dim)


def test_terminal_advantages_do_not_cross_episodes():
    ep1 = [PPOTransition(np.zeros(2), np.zeros(1), 0., 0., 1., False),
           PPOTransition(np.zeros(2), np.zeros(1), 0., 0., 2., True)]
    ep2 = [PPOTransition(np.zeros(2), np.zeros(1), 0., 0., 5., True)]
    _, adv, returns = advantages_from_complete_episodes([ep1, ep2], gamma=1., gae_lambda=1.)
    assert np.allclose(adv, [3., 2., 5.])
    assert np.allclose(returns, [3., 2., 5.])


def test_incomplete_episode_is_rejected():
    ep = [PPOTransition(np.zeros(2), np.zeros(1), 0., 0., 1., False)]
    with pytest.raises(ValueError, match='terminated'):
        advantages_from_complete_episodes([ep])


def test_update_changes_weights():
    torch.manual_seed(0)
    model = FakeModel(3, 2)
    obs = np.array([0.2, -0.1, 0.4], dtype=np.float32)
    act = np.array([0.1, 0.2], dtype=np.float32)
    with torch.no_grad():
        val, logp, _ = model.policy.evaluate_actions(torch.tensor(obs[None]), torch.tensor(act[None]))
    episode = [PPOTransition(obs, act, float(logp.item()), float(val.item()), 2., True)]
    before = [p.clone() for p in model.policy.parameters()]
    result = ppo_update(model, [episode], epochs=2)
    assert result['samples'] == 1
    assert result['updates'] == 2
    assert any(not torch.equal(p, old) for p, old in zip(model.policy.parameters(), before))
