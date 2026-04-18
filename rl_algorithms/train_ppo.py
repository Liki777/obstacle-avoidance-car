#!/usr/bin/env python3
"""
PPO 训练入口（Mock 或 Gazebo）。

示例::

    python -m rl_algorithms.train_ppo --mock --total-updates 30
    python -m rl_algorithms.train_ppo --gazebo --reward-profile stage2_simple_avoid ...
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _wilson_sr_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """二项成功率 Wilson 区间（近似 95% 当 z=1.96）。"""
    if n <= 0:
        return 0.0, 1.0
    k = float(max(0, min(int(successes), int(n))))
    nn = float(int(n))
    phat = k / nn
    denom = 1.0 + z * z / nn
    centre = (phat + z * z / (2.0 * nn)) / denom
    rad = z * math.sqrt((phat * (1.0 - phat) / nn + z * z / (4.0 * nn * nn))) / denom
    return max(0.0, centre - rad), min(1.0, centre + rad)


import numpy as np
try:
    import torch
except ModuleNotFoundError as e:
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None

def _make_gazebo_env(
    *,
    level: int,
    oob_pose_frame: str,
    control_dt: float,
    max_episode_steps: int,
    spawn_x: float,
    spawn_y: float,
    spawn_yaw: float,
    spawn_yaw_towards_goal: bool,
    goal_x: float,
    goal_y: float,
    goal_range_x: str,
    goal_range_y: str,
    goal_min_distance: float,
    map_x_min: float,
    map_x_max: float,
    map_y_min: float,
    map_y_max: float,
    step_log_csv: str | None,
    reward_profile: str,
    use_avoid_course_goals: bool,
    avoid_course_curriculum: bool,
    curriculum_total_episodes: int,
    linear_x_max: float | None,
    angular_z_max: float | None,
    angular_z_min: float | None,
):
    from obstacle_environment import RobotTaskSpec
    from obstacle_environment.gym_env.gym_env import GazeboEnvConfig, RlCarGazeboEnv
    from obstacle_environment.reward import RewardConfig

    if reward_profile == "stage1_walk":
        r_cfg = RewardConfig(
            # Level0：避免「cos(goal_angle) 密集正奖」盖住 progress（车可学会：对准目标但倒车/站桩仍高分）。
            k_progress=1.0,
            k_direction=0.0,
            k_velocity_direction=0.7,
            k_behind_goal=0.25,
            k_safe=0.0,
            k_risk=0.0,
            k_front_safe=0.0,
            k_front_risk=0.0,
            k_front_close_penalty=0.0,
            k_lateral_detour=0.0,
            k_turn=0.08,
            k_stop=0.9,
            k_smooth=0.03,
            # Level0：仍保留碰撞/成功终止项；阈值略放宽减少噪声误触发
            collision_distance=0.12,
            safe_distance=0.55,
            goal_reached_distance=0.45,
            success_reward=50.0,
            collision_penalty=50.0,
            out_of_bounds_penalty=25.0,
        )
    elif reward_profile == "stage2_simple_avoid":
        r_cfg = RewardConfig.stage2_simple_avoid()
    elif reward_profile == "stage3_hard_avoid":
        r_cfg = RewardConfig.stage3_hard_avoid()
    else:
        r_cfg = RewardConfig()

    walk_no_reverse = bool(int(level) == 0 and reward_profile == "stage1_walk")
    spec = RobotTaskSpec.preset_diff_drive(
        reward_config=r_cfg,
        linear_x_min=0.0 if walk_no_reverse else None,
        linear_x_max=linear_x_max,
        angular_z_max=angular_z_max,
        angular_z_min=angular_z_min,
    )

    goal_sample_range = None
    if (not use_avoid_course_goals) and goal_range_x and goal_range_y:
        x0, x1 = [float(s.strip()) for s in goal_range_x.split(",")]
        y0, y1 = [float(s.strip()) for s in goal_range_y.split(",")]
        goal_sample_range = ((x0, x1), (y0, y1))
    env_cfg = GazeboEnvConfig(
        control_dt=control_dt,
        max_episode_steps=max_episode_steps,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_yaw=spawn_yaw,
        spawn_yaw_towards_goal=spawn_yaw_towards_goal,
        goal_x=goal_x,
        goal_y=goal_y,
        goal_sample_range=goal_sample_range,
        goal_min_distance=goal_min_distance,
        map_x_min=map_x_min,
        map_x_max=map_x_max,
        map_y_min=map_y_min,
        map_y_max=map_y_max,
        oob_pose_frame=str(oob_pose_frame),
        done_on_out_of_bounds=True,
        step_log_csv=step_log_csv,
        use_avoid_course_goals=use_avoid_course_goals,
        avoid_course_curriculum=avoid_course_curriculum,
        curriculum_total_episodes=curriculum_total_episodes,
        enable_level1_static_obstacles=bool(int(level) >= 1),
    )
    return RlCarGazeboEnv(spec, env_cfg), spec


def main() -> None:
    if _TORCH_IMPORT_ERROR is not None or torch is None:
        raise SystemExit(
            "缺少依赖 torch。请先执行 `pip install -r requirements.txt`（建议在 venv 内）。"
        )
    from rl_algorithms.envs.mock_car_env import MockCarEnv
    from rl_algorithms.ppo.ppo import PPOConfig, PPOTrainer

    ap = argparse.ArgumentParser(description="PPO 训练（Mock 或 Gazebo）")
    ap.add_argument("--mock", action="store_true", help="使用 MockCarEnv（不连 ROS）")
    ap.add_argument("--gazebo", action="store_true", help="使用 RlCarGazeboEnv")
    ap.add_argument(
        "--level",
        type=int,
        default=0,
        help="课程阶段：0=走路到固定目标；1=静态障碍；2=少量动态；3=多动态（后续扩展）",
    )
    ap.add_argument("--total-updates", type=int, default=50)
    ap.add_argument("--rollout-steps", type=int, default=512)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--save",
        type=str,
        default="",
        help="保存路径（默认 checkpoints/level{N}/latest.pt）",
    )
    ap.add_argument("--load", type=str, default="", help="从 checkpoint 续训")
    ap.add_argument("--auto-resume", action="store_true", help="若 save 已存在则自动加载")
    ap.add_argument(
        "--no-auto-prev-load",
        action="store_true",
        help="禁用“level>0 默认加载上一个 level checkpoint”的行为",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="仅评估：连续跑 N 个完整 episode 后退出（不训练、不写 checkpoint）；需 --load 或 --auto-resume",
    )
    ap.add_argument(
        "--eval-stochastic",
        action="store_true",
        help="评估时对动作采样（默认用确定性均值动作，便于复现）",
    )

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--minibatch-size", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)

    ap.add_argument("--control-dt", type=float, default=0.1)
    ap.add_argument("--max-episode-steps", type=int, default=300)
    ap.add_argument("--spawn-x", type=float, default=0.0)
    ap.add_argument("--spawn-y", type=float, default=0.0)
    ap.add_argument("--spawn-yaw", type=float, default=0.0)
    ap.add_argument(
        "--spawn-yaw-towards-goal",
        action="store_true",
        help="reset 时车头朝向当前采样目标（阶段2/3 推荐）",
    )
    ap.add_argument("--goal-x", type=float, default=4.0)
    ap.add_argument("--goal-y", type=float, default=2.0)
    ap.add_argument("--goal-range-x", type=str, default="")
    ap.add_argument("--goal-range-y", type=str, default="")
    ap.add_argument("--goal-min-distance", type=float, default=0.8)
    ap.add_argument("--map-x-min", type=float, default=-1)
    ap.add_argument("--map-x-max", type=float, default=10.5)
    ap.add_argument("--map-y-min", type=float, default=-1)
    ap.add_argument("--map-y-max", type=float, default=9.5)
    ap.add_argument(
        "--oob-pose-frame",
        type=str,
        default="",
        choices=["", "odom", "world"],
        help="越界判定坐标系：odom 用 /odom.pose；world 用 TF(world->base_link)。留空则 level0 默认 world，其余默认 odom",
    )
    ap.add_argument("--step-log-csv", type=str, default="logs/ppo_steps.csv")
    ap.add_argument(
        "--reward-profile",
        type=str,
        default="",
        choices=[
            "",
            "default",
            "stage1_walk",
            "stage2_simple_avoid",
            "stage3_hard_avoid",
        ],
        help="奖励预设；留空则按 level 选择（level0 默认 stage1_walk，其余默认 default）",
    )
    ap.add_argument(
        "--avoid-course-goals",
        action="store_true",
        help="使用 avoid_training_course 地图配套终点采样（需与地图边界一致）",
    )
    ap.add_argument(
        "--avoid-course-curriculum",
        action="store_true",
        help="终点按 episode 难度曲线采样（配合 --avoid-course-goals）",
    )
    ap.add_argument("--curriculum-total-episodes", type=int, default=50_000)
    ap.add_argument(
        "--linear-x-max",
        type=float,
        default=None,
        help="限制 |linear.x| 上界（默认 1.0）；仅 Gazebo",
    )
    ap.add_argument(
        "--angular-z-max",
        type=float,
        default=None,
        help="限制 angular.z 上界（默认 0.5）；仅指定 max 时 min=-max。需要更快转向时可显式调大",
    )
    ap.add_argument(
        "--angular-z-min",
        type=float,
        default=None,
        help="显式设置 angular.z 下界；未设且未设 --angular-z-max 时为 -0.5",
    )

    args = ap.parse_args()

    if args.mock and args.gazebo:
        print("不能同时指定 --mock 与 --gazebo")
        sys.exit(1)
    if not args.mock and not args.gazebo:
        print("请指定 --mock 或 --gazebo")
        sys.exit(1)

    if (not args.mock) and int(args.max_episode_steps) > int(args.rollout_steps):
        print(
            f"[WARN] max_episode_steps={int(args.max_episode_steps)} > rollout_steps={int(args.rollout_steps)}："
            "单次 rollout 内可能不出现 episode 结束，日志里 ep_done/mean_ep_len 常为 0；"
            "建议令 max_episode_steps <= rollout_steps（例如 256/512）。",
            flush=True,
        )

    # ---- level -> 默认 reward/save/load ----
    level = int(args.level)
    if level < 0:
        level = 0

    if not args.reward_profile:
        # Level0：学会朝固定目标“走路”
        args.reward_profile = "stage1_walk" if level == 0 else "default"

    # Level0：默认地图边界沿用 maze 训练区会在“空旷大地图”上造成频繁 out_of_bounds。
    # 若用户未显式修改边界（仍为默认值），则自动放宽到覆盖 spawn/goal 的合理范围。
    if (not args.mock) and level == 0:
        default_bounds = (-1.0, 10.5, -1.0, 9.5)
        cur_bounds = (float(args.map_x_min), float(args.map_x_max), float(args.map_y_min), float(args.map_y_max))
        if cur_bounds == default_bounds:
            margin = 8.0
            sx, sy = float(args.spawn_x), float(args.spawn_y)
            gx, gy = float(args.goal_x), float(args.goal_y)
            args.map_x_min = min(sx, gx) - margin
            args.map_x_max = max(sx, gx) + margin
            args.map_y_min = min(sy, gy) - margin
            args.map_y_max = max(sy, gy) + margin

    if not args.save:
        args.save = os.path.join(_ROOT, "checkpoints", f"level{level}", "latest.pt")

    # 默认：level>0 自动加载上一级（除非显式 load 或禁用）
    if (not args.load.strip()) and (not bool(args.no_auto_prev_load)) and level > 0:
        prev = os.path.join(_ROOT, "checkpoints", f"level{level-1}", "latest.pt")
        if os.path.exists(prev):
            args.load = prev

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mock:
        env = MockCarEnv()
        obs_dim, act_dim = env.state_dim, env.action_dim
        spec = None
    else:
        # Level0：固定目标（不采样），并建议 reset 时车头朝向目标以加速学到“走路”
        spawn_yaw_towards_goal = bool(args.spawn_yaw_towards_goal)
        if level == 0:
            spawn_yaw_towards_goal = True

        oob_pose_frame = str(args.oob_pose_frame).strip().lower()
        if not oob_pose_frame:
            oob_pose_frame = "world" if int(level) == 0 else "odom"

        env, spec = _make_gazebo_env(
            level=level,
            oob_pose_frame=oob_pose_frame,
            control_dt=args.control_dt,
            max_episode_steps=args.max_episode_steps,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_yaw=args.spawn_yaw,
            spawn_yaw_towards_goal=spawn_yaw_towards_goal,
            goal_x=args.goal_x,
            goal_y=args.goal_y,
            goal_range_x=args.goal_range_x,
            goal_range_y=args.goal_range_y,
            goal_min_distance=args.goal_min_distance,
            map_x_min=args.map_x_min,
            map_x_max=args.map_x_max,
            map_y_min=args.map_y_min,
            map_y_max=args.map_y_max,
            step_log_csv=(args.step_log_csv or None),
            reward_profile=args.reward_profile,
            use_avoid_course_goals=args.avoid_course_goals,
            avoid_course_curriculum=args.avoid_course_curriculum,
            curriculum_total_episodes=args.curriculum_total_episodes,
            linear_x_max=args.linear_x_max,
            angular_z_max=args.angular_z_max,
            angular_z_min=args.angular_z_min,
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
    load_path = args.load.strip()
    if (not load_path) and args.auto_resume and os.path.exists(args.save):
        load_path = args.save

    eval_n = int(args.eval_episodes)
    if eval_n > 0 and (not load_path):
        print(
            "错误：--eval-episodes 需要指定 --load=...，或使用 --auto-resume（且默认 save 路径上已有权重）。",
            flush=True,
        )
        sys.exit(1)

    if load_path:
        trainer.load(load_path)
        print(f"loaded checkpoint: {load_path} (global_update={trainer.global_update})", flush=True)

    if eval_n > 0:
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
            sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass
        t0 = time.time()
        det = not bool(args.eval_stochastic)
        print(
            f"start eval | env={'mock' if args.mock else 'gazebo'} reward_profile={args.reward_profile} "
            f"episodes={eval_n} deterministic={det}",
            flush=True,
        )
        stats = trainer.evaluate_episodes(eval_n, deterministic=det)
        n_ep = int(stats["n"])
        succ_n = int(stats.get("successes", int(round(float(stats["success_rate"]) * max(1, n_ep)))))
        lo, hi = _wilson_sr_ci(succ_n, n_ep)
        print(
            f"[eval] summary | mean_return={stats['mean_return']:.3f} std_return={stats['std_return']:.3f} "
            f"mean_len={stats['mean_len']:.1f} success={succ_n}/{n_ep} ({stats['success_rate']:.1%}) "
            f"Wilson95≈[{lo:.1%},{hi:.1%}] term={stats['term']} | elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        print("", flush=True)
        if not args.mock and hasattr(env, "close"):
            env.close()
        return

    t0 = time.time()
    print(
        f"start training | env={'mock' if args.mock else 'gazebo'} "
        f"reward_profile={args.reward_profile} updates={args.total_updates}",
        flush=True,
    )
    start_u = int(trainer.global_update)
    target_u = start_u + int(args.total_updates)
    for u in range(start_u, target_u):
        t_u = time.time()
        print(f"collecting rollout {u+1}/{target_u} ...", flush=True)
        batch = trainer.collect_rollout()
        mean_r = float(batch.get("mean_step_reward", 0.0))
        mean_abs_r = float(batch.get("mean_abs_step_reward", 0.0))
        episodes_done = int(batch.get("episodes_done", 0))
        mean_ep_len = float(batch.get("mean_ep_len", 0.0))
        rb_prog = float(batch.get("mean_rb_progress", 0.0))
        rb_dir = float(batch.get("mean_rb_direction", 0.0))
        rb_turn = float(batch.get("mean_rb_turn", 0.0))
        rb_term = float(batch.get("mean_rb_terminal", 0.0))
        tc = int(batch.get("term_collision", 0))
        ts = int(batch.get("term_success", 0))
        toob = int(batch.get("term_oob", 0))
        ttr = int(batch.get("term_trunc", 0))

        # 只把张量留给 update()
        tensor_keys = {"obs", "actions", "logprobs", "advantages", "returns", "values"}
        train_batch = {k: batch[k] for k in tensor_keys if k in batch}

        print(f"updating policy {u+1}/{target_u} ...", flush=True)
        stats = trainer.update(train_batch)
        trainer.global_update = int(u + 1)
        print(
            f"update {u+1}/{target_u} | "
            f"mean_step_r={mean_r:.4f} mean_abs_r={mean_abs_r:.4f} | "
            f"rb: prog={rb_prog:+.4f} dir={rb_dir:+.4f} turn={rb_turn:+.4f} term={rb_term:+.4f} | "
            f"ep_done={episodes_done} mean_ep_len={mean_ep_len:.1f} | "
            f"term{{col={tc} succ={ts} oob={toob} trunc={ttr}}} | "
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
