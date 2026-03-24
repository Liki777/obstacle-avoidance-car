"""连续动作 Actor-Critic：高斯 + tanh 压到 (-1,1)。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def _squashed_normal_logprob(
        mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        a = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        x_t = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + eps)
        log_prob = log_prob.sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std.expand_as(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = mean if deterministic else dist.rsample()
        action = torch.tanh(x_t)  # 均值路径也经 tanh，落在 (-1,1)
        log_prob, _ = self._squashed_normal_logprob(mean, std, action)
        value = self.get_value(obs)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std.expand_as(mean)
        std = log_std.exp()
        log_prob, entropy = self._squashed_normal_logprob(mean, std, action)
        value = self.get_value(obs)
        return log_prob, entropy, value
