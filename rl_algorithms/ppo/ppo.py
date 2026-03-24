"""PPO (clip) + GAE，适用于 ``MockCarEnv`` 与 ``RlCarGazeboEnv``。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.ppo.networks import ActorCritic


class SteppableEnv(Protocol):
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]: ...


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    num_epochs: int = 10
    minibatch_size: int = 64
    rollout_steps: int = 2048


class PPOTrainer:
    def __init__(
        self,
        env: SteppableEnv,
        *,
        obs_dim: int,
        act_dim: int,
        device: str | torch.device = "cpu",
        config: PPOConfig | None = None,
        hidden_dim: int = 256,
    ) -> None:
        self.env = env
        self.cfg = config or PPOConfig()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = ActorCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - float(dones[t])
            next_val = values[t + 1] if t + 1 < T else last_value
            delta = rewards[t] + self.cfg.gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + values[:T]
        return adv, ret

    def collect_rollout(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        obs_buf = np.zeros((cfg.rollout_steps, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, self.act_dim), dtype=np.float32)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps + 1,), dtype=np.float32)

        o = self.env.reset()
        ep_ret = 0.0
        ep_len = 0

        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, logp, v = self.net.act(obs_t)
                a = action.cpu().numpy().reshape(-1)
                logp_np = float(logp.cpu().numpy().item())
                v_np = float(v.cpu().numpy().item())

                obs_buf[t] = o
                act_buf[t] = a
                logp_buf[t] = logp_np
                val_buf[t] = v_np

                o, r, done, _ = self.env.step(a)
                rew_buf[t] = float(r)
                done_buf[t] = float(done)
                ep_ret += float(r)
                ep_len += 1

                if done:
                    o = self.env.reset()
                    ep_ret = 0.0
                    ep_len = 0

            obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, _, v_last = self.net.act(obs_t)
            val_buf[cfg.rollout_steps] = float(v_last.cpu().numpy().item())

        adv, ret = self._compute_gae(rew_buf, val_buf[:-1], done_buf, float(val_buf[-1]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "obs": obs_buf,
            "actions": act_buf,
            "logprobs": logp_buf,
            "advantages": adv.astype(np.float32),
            "returns": ret.astype(np.float32),
            "values": val_buf[:-1].astype(np.float32),
        }

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        logp_old = torch.as_tensor(batch["logprobs"], device=self.device)
        adv = torch.as_tensor(batch["advantages"], device=self.device)
        ret = torch.as_tensor(batch["returns"], device=self.device)

        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        updates = 0

        for _ in range(cfg.num_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if mb.size < 2:
                    continue
                ob = obs[mb]
                ac = actions[mb]
                lp_old = logp_old[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]

                logp, ent, v = self.net.evaluate(ob, ac)
                ratio = torch.exp(logp - lp_old)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_b
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = 0.5 * ((v - ret_b) ** 2).mean()
                ent_loss = -ent.mean()
                loss = pol_loss + cfg.value_coef * val_loss + cfg.entropy_coef * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (lp_old - logp).mean().item()
                stats["policy_loss"] += float(pol_loss.item())
                stats["value_loss"] += float(val_loss.item())
                stats["entropy"] += float((-ent_loss).item())
                stats["approx_kl"] += approx_kl
                updates += 1

        if updates > 0:
            for k in stats:
                stats[k] /= updates
        return stats

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "obs_dim": self.obs_dim, "act_dim": self.act_dim}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
