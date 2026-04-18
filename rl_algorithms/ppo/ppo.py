"""PPO (clip) + GAE，适用于 ``MockCarEnv`` 与 ``RlCarGazeboEnv``。"""

from __future__ import annotations

import sys
from collections import defaultdict
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
    entropy_coef: float = 0.05
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
        self.global_update: int = 0

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

    def collect_rollout(self) -> dict[str, Any]:
        cfg = self.cfg
        obs_buf = np.zeros((cfg.rollout_steps, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, self.act_dim), dtype=np.float32)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps + 1,), dtype=np.float32)

        # ---- rollout diagnostics（用于定位“原地转圈/频繁 reset”）----
        rb_sum = {
            "progress": 0.0,
            "direction": 0.0,
            "safe": 0.0,
            "risk": 0.0,
            "turn": 0.0,
            "stop": 0.0,
            "smooth": 0.0,
            "front_safe": 0.0,
            "front_risk": 0.0,
            "terminal": 0.0,
        }
        term_counts: dict[str, int] = {
            "": 0,
            "collision": 0,
            "success": 0,
            "out_of_bounds": 0,
            "truncated": 0,
            "other": 0,
        }
        ep_done = 0
        ep_lens: list[int] = []
        cur_ep_len = 0

        # PPO update() 期间主线程不 spin，DDS 回调积压；下一轮首个 reset 易误判超时
        if hasattr(self.env, "spin_ros"):
            self.env.spin_ros(180)

        o = self.env.reset()
        ep_ret = 0.0
        ep_len = 0
        cur_ep_len = 0

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

                o, r, done, info = self.env.step(a)
                rew_buf[t] = float(r)
                done_buf[t] = float(done)
                ep_ret += float(r)
                ep_len += 1
                cur_ep_len += 1

                rb = info.get("reward_breakdown") if isinstance(info, dict) else None
                if rb is not None:
                    rb_sum["progress"] += float(getattr(rb, "progress", 0.0))
                    rb_sum["direction"] += float(getattr(rb, "direction", 0.0))
                    rb_sum["safe"] += float(getattr(rb, "safe", 0.0))
                    rb_sum["risk"] += float(getattr(rb, "risk", 0.0))
                    rb_sum["turn"] += float(getattr(rb, "turn", 0.0))
                    rb_sum["stop"] += float(getattr(rb, "stop", 0.0))
                    rb_sum["smooth"] += float(getattr(rb, "smooth", 0.0))
                    rb_sum["front_safe"] += float(getattr(rb, "front_safe", 0.0))
                    rb_sum["front_risk"] += float(getattr(rb, "front_risk", 0.0))
                    rb_sum["terminal"] += float(getattr(rb, "terminal", 0.0))

                if done:
                    terminated = bool(info.get("terminated", False)) if isinstance(info, dict) else False
                    truncated = bool(info.get("truncated", False)) if isinstance(info, dict) else False
                    reason = str(info.get("terminal_reason", "")) if isinstance(info, dict) else ""
                    if truncated and (not terminated):
                        term_counts["truncated"] += 1
                    else:
                        if reason in term_counts:
                            term_counts[reason] += 1
                        elif reason:
                            term_counts["other"] += 1
                        else:
                            term_counts[""] += 1

                    ep_done += 1
                    ep_lens.append(int(cur_ep_len))
                    cur_ep_len = 0

                    o = self.env.reset()
                    ep_ret = 0.0
                    ep_len = 0

            obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, _, v_last = self.net.act(obs_t)
            val_buf[cfg.rollout_steps] = float(v_last.cpu().numpy().item())

        adv, ret = self._compute_gae(rew_buf, val_buf[:-1], done_buf, float(val_buf[-1]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = float(cfg.rollout_steps)
        diag: dict[str, float | int] = {
            "mean_step_reward": float(np.mean(rew_buf)),
            "mean_abs_step_reward": float(np.mean(np.abs(rew_buf))),
            "episodes_done": int(ep_done),
            "mean_rb_progress": float(rb_sum["progress"] / T),
            "mean_rb_direction": float(rb_sum["direction"] / T),
            "mean_rb_safe": float(rb_sum["safe"] / T),
            "mean_rb_risk": float(rb_sum["risk"] / T),
            "mean_rb_turn": float(rb_sum["turn"] / T),
            "mean_rb_stop": float(rb_sum["stop"] / T),
            "mean_rb_smooth": float(rb_sum["smooth"] / T),
            "mean_rb_front_safe": float(rb_sum["front_safe"] / T),
            "mean_rb_front_risk": float(rb_sum["front_risk"] / T),
            "mean_rb_terminal": float(rb_sum["terminal"] / T),
            "term_collision": int(term_counts["collision"]),
            "term_success": int(term_counts["success"]),
            "term_oob": int(term_counts["out_of_bounds"]),
            "term_trunc": int(term_counts["truncated"]),
            "term_other": int(term_counts["other"] + term_counts[""]),
        }
        if ep_lens:
            diag["mean_ep_len"] = float(sum(ep_lens) / max(1, len(ep_lens)))

        out: dict[str, np.ndarray | float | int] = {
            "obs": obs_buf,
            "actions": act_buf,
            "logprobs": logp_buf,
            "advantages": adv.astype(np.float32),
            "returns": ret.astype(np.float32),
            "values": val_buf[:-1].astype(np.float32),
        }
        out.update(diag)  # type: ignore[arg-type]
        return out  # type: ignore[return-value]

    def evaluate_episodes(self, n_episodes: int, *, deterministic: bool = True) -> dict[str, Any]:
        """
        跑若干完整 episode（不更新策略），用于肉眼看仿真或统计成功率/回报。

        默认 ``deterministic=True``：actor 输出经 tanh 的**均值动作**（与训练时采样不同）。
        """
        n_episodes = int(n_episodes)
        if n_episodes <= 0:
            return {
                "n": 0,
                "mean_return": 0.0,
                "std_return": 0.0,
                "mean_len": 0.0,
                "success_rate": 0.0,
                "successes": 0,
                "term": {},
            }

        # 减少与其它进程/ROS 日志交错时「上一行末尾粘下一行」的现象（仍可能被 rclpy 打断）。
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
            sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass

        if hasattr(self.env, "spin_ros"):
            self.env.spin_ros(180)

        step_cap = 1_000_000
        if hasattr(self.env, "env_cfg") and hasattr(self.env.env_cfg, "max_episode_steps"):
            step_cap = int(self.env.env_cfg.max_episode_steps) * 3 + 200
        elif hasattr(self.env, "max_steps"):
            step_cap = int(self.env.max_steps) * 3 + 200

        returns: list[float] = []
        lengths: list[int] = []
        term: dict[str, int] = defaultdict(int)
        successes = 0

        for ep in range(n_episodes):
            o = self.env.reset()
            ep_ret = 0.0
            steps = 0
            last_reason = ""
            while steps < step_cap:
                with torch.no_grad():
                    obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _ = self.net.act(obs_t, deterministic=deterministic)
                    a = action.cpu().numpy().reshape(-1)
                o, r, done, info = self.env.step(a)
                ep_ret += float(r)
                steps += 1
                if isinstance(info, dict):
                    last_reason = str(info.get("terminal_reason", "") or "")
                if done:
                    if last_reason == "success":
                        successes += 1
                    elif isinstance(info, dict) and float(info.get("mock_norm", 1e9)) < 0.15:
                        successes += 1
                        last_reason = "success"
                    if last_reason:
                        term[last_reason] += 1
                    elif isinstance(info, dict) and bool(info.get("truncated", False)):
                        term["truncated"] += 1
                    else:
                        term["unknown"] += 1
                    disp = last_reason or (
                        "truncated"
                        if isinstance(info, dict) and bool(info.get("truncated", False))
                        else "unknown"
                    )
                    returns.append(ep_ret)
                    lengths.append(steps)
                    print(
                        f"[eval] episode {ep + 1}/{n_episodes} | return={ep_ret:.3f} steps={steps} reason={disp}",
                        flush=True,
                    )
                    break
            else:
                term["step_cap"] += 1
                returns.append(ep_ret)
                lengths.append(steps)
                print(
                    f"[eval] episode {ep + 1}/{n_episodes} | return={ep_ret:.3f} steps={steps} reason=step_cap",
                    flush=True,
                )

        mean_r = float(np.mean(returns)) if returns else 0.0
        std_r = float(np.std(returns)) if returns else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        sr = float(successes) / float(max(1, n_episodes))
        return {
            "n": n_episodes,
            "mean_return": mean_r,
            "std_return": std_r,
            "mean_len": mean_len,
            "success_rate": sr,
            "successes": int(successes),
            "term": dict(term),
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
        torch.save(
            {
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "global_update": int(self.global_update),
                "cfg": {
                    "gamma": float(self.cfg.gamma),
                    "gae_lambda": float(self.cfg.gae_lambda),
                    "clip_coef": float(self.cfg.clip_coef),
                    "value_coef": float(self.cfg.value_coef),
                    "entropy_coef": float(self.cfg.entropy_coef),
                    "max_grad_norm": float(self.cfg.max_grad_norm),
                    "lr": float(self.cfg.lr),
                    "num_epochs": int(self.cfg.num_epochs),
                    "minibatch_size": int(self.cfg.minibatch_size),
                    "rollout_steps": int(self.cfg.rollout_steps),
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "opt" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt"])
            except Exception:
                # 允许只加载网络参数（例如你改了 optimizer 超参）
                pass
        self.global_update = int(ckpt.get("global_update", 0))
