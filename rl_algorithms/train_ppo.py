#!/usr/bin/env python3
"""
PPO 训练入口。

无仿真冒烟::
    cd graduation_project && python -m rl_algorithms.train_ppo --mock --total-updates 30

Gazebo（需另开终端已 launch 仿真并 source install/setup.bash）::
    python -m rl_algorithms.train_ppo --gazebo --total-updates 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 项目根目录加入 path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from rl_algorithms.envs.mock_car_env import MockCarEnv
from rl_algorithms.ppo.ppo import PPOConfig, PPOTrainer


def _make_gazebo_env(
    *,
    control_dt: float,
    max_episode_steps: int,
    spawn_x: float,
    spawn_y: float,
    spawn_yaw: float,
    goal_x: float,
    goal_y: float,
    map_x_min: float,
    map_x_max: float,
    map_y_min: float,
    map_y_max: float,
):
    from obstacle_environment import RobotTaskSpec
    from obstacle_environment.gym_env.gym_env import GazeboEnvConfig, RlCarGazeboEnv

    spec = RobotTaskSpec.preset_diff_drive()
    env_cfg = GazeboEnvConfig(
        control_dt=control_dt,
        max_episode_steps=max_episode_steps,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_yaw=spawn_yaw,
        goal_x=goal_x,
        goal_y=goal_y,
        map_x_min=map_x_min,
        map_x_max=map_x_max,
        map_y_min=map_y_min,
        map_y_max=map_y_max,
        done_on_out_of_bounds=True,
    )
    return RlCarGazeboEnv(spec, env_cfg), spec


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO 训练（Mock 或 Gazebo）")
    ap.add_argument("--mock", action="store_true", help="使用 MockCarEnv（不连 ROS）")
    ap.add_argument("--gazebo", action="store_true", help="使用 RlCarGazeboEnv")
    ap.add_argument("--total-updates", type=int, default=50, help="策略更新轮数（每轮一次 rollout + 多 epoch 优化）")
    ap.add_argument("--rollout-steps", type=int, default=512, help="每轮采集的转移步数")
    ap.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    ap.add_argument("--save", type=str, default=os.path.join(_ROOT, "checkpoints", "ppo_car.pt"))
    ap.add_argument("--seed", type=int, default=0)

    # ---------- PPO / 网络超参（与 PPOConfig、ActorCritic 一致）----------
    ap.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE λ")
    ap.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip 范围系数")
    ap.add_argument("--value-coef", type=float, default=0.5, help="价值损失权重")
    ap.add_argument("--entropy-coef", type=float, default=0.01, help="策略熵 bonus 权重（鼓励探索）")
    ap.add_argument("--max-grad-norm", type=float, default=0.5, help="梯度裁剪")
    ap.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率")
    ap.add_argument("--ppo-epochs", type=int, default=10, help="每轮 rollout 后优化 epoch 数")
    ap.add_argument("--minibatch-size", type=int, default=64, help="小批量大小")
    ap.add_argument("--hidden-dim", type=int, default=256, help="Actor/Critic MLP 隐层宽度")

    # ---------- 仅 Gazebo 环境 ----------
    ap.add_argument("--control-dt", type=float, default=0.1, help="每步 env 内等待时间 (s)")
    ap.add_argument("--max-episode-steps", type=int, default=256, help="仿真回合最大步数（截断）")
    ap.add_argument("--spawn-x", type=float, default=0.0)
    ap.add_argument("--spawn-y", type=float, default=0.0)
    ap.add_argument("--spawn-yaw", type=float, default=0.0)
    ap.add_argument("--goal-x", type=float, default=2.0)
    ap.add_argument("--goal-y", type=float, default=2.0)
    ap.add_argument("--map-x-min", type=float, default=-2.8)
    ap.add_argument("--map-x-max", type=float, default=2.8)
    ap.add_argument("--map-y-min", type=float, default=-2.8)
    ap.add_argument("--map-y-max", type=float, default=2.8)

    args = ap.parse_args()

    if args.mock and args.gazebo:
        print("不能同时指定 --mock 与 --gazebo")
        sys.exit(1)
    if not args.mock and not args.gazebo:
        print("请指定 --mock 或 --gazebo")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mock:
        env = MockCarEnv()
        obs_dim, act_dim = env.state_dim, env.action_dim
        spec = None
    else:
        env, spec = _make_gazebo_env(
            control_dt=args.control_dt,
            max_episode_steps=args.max_episode_steps,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_yaw=args.spawn_yaw,
            goal_x=args.goal_x,
            goal_y=args.goal_y,
            map_x_min=args.map_x_min,
            map_x_max=args.map_x_max,
            map_y_min=args.map_y_min,
            map_y_max=args.map_y_max,
        )
        obs_dim = spec.state_dim
        act_dim = spec.action_dim

    cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        num_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=args.rollout_steps,
    )
    trainer = PPOTrainer(
        env,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=args.device,
        config=cfg,
        hidden_dim=args.hidden_dim,
    )

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    t0 = time.time()
    print(
        f"start training | env={'mock' if args.mock else 'gazebo'} "
        f"updates={args.total_updates} rollout_steps={args.rollout_steps} "
        f"control_dt={args.control_dt if args.gazebo else 'n/a'}",
        flush=True,
    )
    for u in range(args.total_updates):
        t_u = time.time()
        print(f"collecting rollout {u+1}/{args.total_updates} ...", flush=True)
        batch = trainer.collect_rollout()
        print(f"updating policy {u+1}/{args.total_updates} ...", flush=True)
        stats = trainer.update(batch)
        print(
            f"update {u+1}/{args.total_updates} | "
            f"pol_loss={stats['policy_loss']:.4f} val_loss={stats['value_loss']:.4f} "
            f"H={stats['entropy']:.3f} kl~={stats['approx_kl']:.4f} "
            f"elapsed={time.time()-t_u:.1f}s",
            flush=True,
        )

    trainer.save(args.save)
    print(f"saved {args.save} in {time.time()-t0:.1f}s", flush=True)

    if not args.mock and hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
