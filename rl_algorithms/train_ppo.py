#!/usr/bin/env python3
"""
PPO 训练入口（Mock 或 Gazebo）。

示例::

    python -m rl_algorithms.train_ppo --mock --total-updates 30
    python -m rl_algorithms.train_ppo --gazebo --level 2 ...
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
import shutil
import socket
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _checkpoint_abs(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def _next_rotating_checkpoint_path(save_pt: str) -> str:
    """与 ``latest.pt`` 同目录下递增 ``checkpoint1.pt``, ``checkpoint2.pt``, …"""
    d = os.path.dirname(_checkpoint_abs(save_pt)) or "."
    pat = re.compile(r"^checkpoint(\d+)\.pt$")
    mx = 0
    try:
        for fn in os.listdir(d):
            m = pat.match(fn)
            if m:
                mx = max(mx, int(m.group(1)))
    except FileNotFoundError:
        pass
    return os.path.join(d, f"checkpoint{mx + 1}.pt")


def _backup_save_path_to_rotating(save_pt: str) -> str | None:
    """若 save_pt 存在则复制为下一个 ``checkpoint{N}.pt``，返回备份路径。"""
    ap = _checkpoint_abs(save_pt)
    if not os.path.isfile(ap):
        return None
    dst = _next_rotating_checkpoint_path(ap)
    shutil.copy2(ap, dst)
    return dst


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
    debug_reset: bool,
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
    road_map: str,
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
    elif reward_profile == "stage2_level1_corridor":
        r_cfg = RewardConfig.stage2_level1_corridor()
    elif reward_profile == "stage0_road_follow":
        r_cfg = RewardConfig.stage0_road_follow()
    elif reward_profile == "stage2_simple_avoid":
        r_cfg = RewardConfig.stage2_simple_avoid()
    elif reward_profile == "stage3_hard_avoid":
        r_cfg = RewardConfig.stage3_hard_avoid()
    else:
        r_cfg = RewardConfig()

    walk_no_reverse = bool(int(level) == 0 and reward_profile == "stage1_walk")
    include_road = bool(str(road_map).strip())
    road_no_reverse = bool(include_road or reward_profile == "stage0_road_follow")
    spec = RobotTaskSpec.preset_diff_drive(
        reward_config=r_cfg,
        linear_x_min=0.0 if (walk_no_reverse or road_no_reverse) else None,
        linear_x_max=linear_x_max,
        angular_z_max=angular_z_max,
        angular_z_min=angular_z_min,
        include_road=include_road,
        road_lookahead_n=5,
    )

    # ---- 课程 Level0~5：静态 / 动态障碍模式（与 map_design_document.md 一致）----
    lev = max(0, min(5, int(level)))
    static_mode = "none"
    dyn_mode = "none"
    fixed_static: tuple[tuple[float, float, float], ...] = ()
    dyn_min, dyn_max = 1, 2
    if lev == 1:
        static_mode = "fixed"
        # 空元组：由 gym_env 使用 builtin_level1_fixed_mixed_3x3（3×3 混合障碍）
        fixed_static = ()
    elif lev == 2:
        static_mode = "random"
    elif lev == 3:
        dyn_mode = "fixed"
    elif lev == 4:
        dyn_mode = "random"
        dyn_min, dyn_max = 1, 3
    elif lev == 5:
        static_mode = "random"
        dyn_mode = "random"
        dyn_min, dyn_max = 1, 2

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
        debug_reset=bool(debug_reset),
        spawn_yaw_towards_goal=spawn_yaw_towards_goal,
        road_map_yaml=str(road_map).strip(),
        goal_x=goal_x,
        goal_y=goal_y,
        goal_sample_range=goal_sample_range,
        goal_min_distance=goal_min_distance,
        map_x_min=map_x_min,
        map_x_max=map_x_max,
        map_y_min=map_y_min,
        map_y_max=map_y_max,
        oob_pose_frame=str(oob_pose_frame),
        # 道路任务：goal/marker 坐标使用 Gazebo world，避免 odom 与 world 不一致导致“看着没到绿圈却 success/距离很小”
        goal_frame_id=("world" if include_road else "odom"),
        # 道路任务优先用 out_of_road 控制“离开路面”，禁用 out_of_bounds 以避免 world/odom 坐标系不一致带来的误判。
        done_on_out_of_bounds=(not include_road),
        # 道路任务：success 仅当 Gazebo world 下车体进入 goal 邻域（与绿圈 marker 对齐），不用 road_s 提前判 success。
        done_on_goal=(not include_road),
        done_on_road_end=(not include_road),
        done_on_world_goal=bool(include_road),
        world_goal_success_radius_m=0.40,
        step_log_csv=step_log_csv,
        use_avoid_course_goals=use_avoid_course_goals,
        avoid_course_curriculum=avoid_course_curriculum,
        curriculum_total_episodes=curriculum_total_episodes,
        static_obstacle_mode=static_mode,
        fixed_static_obstacles_xyyaw=fixed_static,
        dynamic_obstacle_mode=dyn_mode,
        dynamic_obstacle_count_min=dyn_min,
        dynamic_obstacle_count_max=dyn_max,
        enable_level1_static_obstacles=False,
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
        help="课程 0~5：0 无障碍到目标；1 固定静态；2 随机静态；3 固定动态；4 随机动态；5 随机静+动（详见 map_design_document.md）",
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
    ap.add_argument(
        "--auto-resume",
        action="store_true",
        help="若未指定 --load 且 --save 已存在，则优先从该路径（通常为 latest.pt）加载续训；"
        "若 load 与 save 为同一文件，开训前先将盘上旧文件备份为同目录 checkpoint{N}.pt",
    )
    ap.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=5,
        help="每完成 N 次 PPO update 写入一次 checkpoint（默认 1，崩溃/kill 后可配合 --auto-resume）；"
        "0 表示本轮循环内不写周期文件，仅在退出时 finally 抢救保存一次",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Gazebo 并行采样环境数（>1 启用多进程并行）；Mock 下会忽略",
    )
    ap.add_argument(
        "--parallel-base-ros-domain-id",
        type=int,
        default=10,
        help="并行采样时的 ROS_DOMAIN_ID 起始值（每个 env +1）",
    )
    ap.add_argument(
        "--parallel-base-gazebo-master-port",
        type=int,
        default=11345,
        help="并行采样时的 GAZEBO_MASTER_URI 端口起始值（每个 env +1）",
    )
    ap.add_argument(
        "--parallel-world",
        type=str,
        default="",
        help="并行采样时启动的 world（默认按 show_map 的 WORLD_BY_LEVEL[level]；可填 level1_arena_8x8_fast.world）",
    )
    ap.add_argument(
        "--parallel-log-dir",
        type=str,
        default="logs/parallel_workers",
        help="并行采样时每个 worker 的 ros2 launch 日志目录（每个 worker 一个 worker_{id}.log）",
    )
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
    ap.add_argument(
        "--debug-reset",
        action="store_true",
        help="打印每次 reset 的重置路径与位姿偏差（用于验证 reset 是否真的生效；会刷屏）",
    )
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
            "stage0_road_follow",
            "stage2_level1_corridor",
            "stage2_simple_avoid",
            "stage3_hard_avoid",
        ],
        help="奖励预设；留空则 level0=stage1_walk，level1=stage2_level1_corridor，level2=stage2_simple_avoid，level3~5=stage3_hard_avoid",
    )
    ap.add_argument(
        "--road-map",
        type=str,
        default="",
        help="道路中心线 YAML（Level0 道路跟随用）。设置后将把道路特征拼接到观测，并用 Δs/cte/heading 奖励。",
    )
    ap.add_argument(
        "--no-arena-8x8",
        action="store_true",
        help="禁用 level1~5 时自动使用 8×8 场地默认边界与 spawn(1,1)/goal(7,7)",
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
    raw_level = int(args.level)
    if raw_level > 5:
        print(f"[WARN] --level={raw_level} 超出课程范围，将按 5（Level5）处理。", flush=True)
    level = max(0, min(5, raw_level))

    if not args.reward_profile:
        if level == 0:
            args.reward_profile = "stage1_walk"
        elif level == 1:
            args.reward_profile = "stage2_level1_corridor"
        elif level == 2:
            args.reward_profile = "stage2_simple_avoid"
        else:
            args.reward_profile = "stage3_hard_avoid"

    # Level1~5：默认 8×8 课程场地（可用 --no-arena-8x8 关闭）
    if (not args.mock) and 1 <= level <= 5 and (not bool(args.no_arena_8x8)):
        args.map_x_min = -0.25
        args.map_x_max = 8.25
        args.map_y_min = -0.25
        args.map_y_max = 8.25
        args.spawn_x = 1.0
        args.spawn_y = 1.0
        args.goal_x = 7.0
        args.goal_y = 7.0
        setattr(args, "spawn_yaw_towards_goal", True)

    # Level0：默认地图边界沿用 maze 训练区会在“空旷大地图”上造成频繁 out_of_bounds。
    # 若用户未显式修改边界（仍为默认值），则自动放宽到覆盖 spawn/goal 的合理范围。
    if (not args.mock) and level == 0:
        # 若指定 --road-map，则优先使用 YAML 内的 spawn/goal/bounds，避免“车生成到路外/墙外”。
        rm = str(getattr(args, "road_map", "") or "").strip()
        if rm:
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(Path(rm).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    sp = data.get("spawn", {}) if isinstance(data.get("spawn"), dict) else {}
                    gl = data.get("goal", {}) if isinstance(data.get("goal"), dict) else {}
                    mp = data.get("map", {}) if isinstance(data.get("map"), dict) else {}
                    # 只有当用户未显式覆盖（仍为默认值）时，才从 YAML 注入
                    if float(args.spawn_x) == 0.0 and float(args.spawn_y) == 0.0:
                        args.spawn_x = float(sp.get("x", args.spawn_x))
                        args.spawn_y = float(sp.get("y", args.spawn_y))
                        args.spawn_yaw = float(sp.get("yaw", args.spawn_yaw))
                    if float(args.goal_x) == 4.0 and float(args.goal_y) == 2.0:
                        args.goal_x = float(gl.get("x", args.goal_x))
                        args.goal_y = float(gl.get("y", args.goal_y))
                    # 道路跟随：不需要“朝目标重置车头”，优先尊重 YAML 里的 yaw
                    setattr(args, "spawn_yaw_towards_goal", False)
                    # bounds：如果仍是默认值，则用 YAML map bounds
                    default_bounds = (-1.0, 10.5, -1.0, 9.5)
                    cur_bounds = (
                        float(args.map_x_min),
                        float(args.map_x_max),
                        float(args.map_y_min),
                        float(args.map_y_max),
                    )
                    if cur_bounds == default_bounds and mp:
                        args.map_x_min = float(mp.get("x_min", args.map_x_min))
                        args.map_x_max = float(mp.get("x_max", args.map_x_max))
                        args.map_y_min = float(mp.get("y_min", args.map_y_min))
                        args.map_y_max = float(mp.get("y_max", args.map_y_max))
            except Exception as e:
                print(f"[WARN] 读取 --road-map 失败（将使用命令行 spawn/goal/bounds）：{e}", flush=True)

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
            debug_reset=bool(getattr(args, "debug_reset", False)),
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
            road_map=str(getattr(args, "road_map", "") or ""),
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
    # auto-resume：在未显式 --load 时优先从 --save（通常为 latest.pt）继续
    if (not load_path) and args.auto_resume and os.path.exists(args.save):
        load_path = args.save

    save_abs = _checkpoint_abs(args.save)
    load_abs = _checkpoint_abs(load_path) if load_path else ""
    same_load_save = bool(load_path) and load_abs == save_abs
    # 每个写盘周期都覆盖 --save；「续训同一 latest」体现在：load 与 save 同路径时先将盘上旧文件备份为 checkpoint{N}.pt（见开训备份）

    eval_n = int(args.eval_episodes)
    if eval_n > 0 and (not load_path):
        print(
            "错误：--eval-episodes 需要指定 --load=...，或使用 --auto-resume（且默认 save 路径上已有权重）。",
            flush=True,
        )
        sys.exit(1)

    def _checkpoint_compatible(path: str) -> tuple[bool, str]:
        """
        返回 (ok, reason)。用于在 obs_dim/act_dim 变化时避免 auto-resume 崩溃。
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
            obs_ck = int(ckpt.get("obs_dim", -1))
            act_ck = int(ckpt.get("act_dim", -1))
            if obs_ck != int(obs_dim) or act_ck != int(act_dim):
                return (
                    False,
                    f"checkpoint dim mismatch: ckpt(obs_dim={obs_ck},act_dim={act_ck}) "
                    f"!= current(obs_dim={int(obs_dim)},act_dim={int(act_dim)})",
                )
            return True, ""
        except Exception as e:
            return False, f"checkpoint inspect failed: {e}"

    if load_path:
        ok, reason = _checkpoint_compatible(load_path)
        if not ok:
            # 如果是 auto-resume 选中的旧 latest（最常见：你改了观测维度），就跳过加载从头训
            if args.auto_resume and (not args.load.strip()) and os.path.abspath(load_path) == os.path.abspath(args.save):
                print(f"[checkpoint] auto-resume skipped incompatible save: {reason}", flush=True)
                load_path = ""
            else:
                raise SystemExit(
                    f"无法加载 checkpoint（与当前观测/动作维度不兼容）：{reason}\n"
                    "提示：你可能修改了观测（例如启用道路特征）导致 obs_dim 变化。"
                    "请改用对应维度重新训练，或去掉 --load/--auto-resume 以从头训练。"
                )
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
        try:
            rd = os.environ.get("ROS_DOMAIN_ID", "")
            gm = os.environ.get("GAZEBO_MASTER_URI", "")
            print(f"eval runtime | ROS_DOMAIN_ID={rd or '(unset)'} GAZEBO_MASTER_URI={gm or '(unset)'}", flush=True)
        except Exception:
            pass
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
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    print(
        f"start training | env={'mock' if args.mock else 'gazebo'} level={level} "
        f"reward_profile={args.reward_profile} updates={args.total_updates}",
        flush=True,
    )
    print(
        f"ppo cfg | gamma={args.gamma} gae_lambda={args.gae_lambda} lr={args.lr} "
        f"rollout_steps={args.rollout_steps} max_episode_steps={args.max_episode_steps} "
        f"ppo_epochs={args.ppo_epochs} minibatch={args.minibatch_size} "
        f"clip={args.clip_coef} value_coef={args.value_coef} entropy_coef={args.entropy_coef} "
        f"hidden_dim={args.hidden_dim} device={args.device}",
        flush=True,
    )
    print(
        f"checkpoint | save={args.save} load={load_path or '(none)'} "
        f"same_load_save={same_load_save} periodic_overwrite_save=True "
        f"global_update_start={trainer.global_update}",
        flush=True,
    )

    if same_load_save:
        bk = _backup_save_path_to_rotating(args.save)
        if bk:
            print(f"[checkpoint] 续训同路径：已将盘上原权重备份为 {bk}", flush=True)

    start_u = int(trainer.global_update)
    target_u = start_u + int(args.total_updates)
    cp_every = max(0, int(args.checkpoint_every_updates))
    save_exc: BaseException | None = None
    try:
        # ---- 并行 Gazebo 采样（多进程）----
        num_envs = max(1, int(args.num_envs))
        use_parallel = (not args.mock) and num_envs > 1
        sampler = None
        if use_parallel:
            from curriculum_maps import WORLD_BY_LEVEL, clamp_level
            from rl_algorithms.ppo.parallel_gazebo_sampler import ParallelGazeboConfig, ParallelGazeboSampler

            world = str(args.parallel_world).strip()
            if not world:
                world = str(WORLD_BY_LEVEL[clamp_level(int(level))])

            # 预检端口占用：若 base gazebo master 端口已被其它 gzserver/show 占用，会导致 worker0 永远收不到 /scan /odom。
            # 自动右移到一个更安全的端口段，避免用户每次手动改参数。
            def _port_free(p: int) -> bool:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", int(p)))
                    s.close()
                    return True
                except OSError:
                    return False

            base_port = int(args.parallel_base_gazebo_master_port)
            if not _port_free(base_port):
                # 尝试几个常用备选段
                for cand in (11445, 12445, 13445, 14445):
                    if _port_free(int(cand)):
                        print(
                            f"[WARN] 并行 base_gazebo_master_port={base_port} 已被占用；自动改用 {cand}（worker 将使用 {cand}~{cand + num_envs - 1}）",
                            flush=True,
                        )
                        base_port = int(cand)
                        break
            pcfg = ParallelGazeboConfig(
                num_envs=num_envs,
                base_ros_domain_id=int(args.parallel_base_ros_domain_id),
                base_gazebo_master_port=int(base_port),
                world=world,
                gui=False,
                server=True,
                use_sim_time=True,
                log_dir=str(args.parallel_log_dir),
            )
            # 让 worker 复用与主进程一致的 GazeboEnvConfig（除 ROS/Gazebo 隔离由 env vars 完成）
            goal_sample_range = None
            if (not bool(args.avoid_course_goals)) and str(args.goal_range_x).strip() and str(args.goal_range_y).strip():
                x0, x1 = [float(s.strip()) for s in str(args.goal_range_x).split(",")]
                y0, y1 = [float(s.strip()) for s in str(args.goal_range_y).split(",")]
                goal_sample_range = ((x0, x1), (y0, y1))

            # 关键：并行 worker 需要与 make_gazebo_env(level=...) 一致的障碍课程配置；
            # 否则会出现“并行训练在空场学会了，eval/单进程在有障碍场地全撞”的错位。
            lev = int(clamp_level(int(level)))
            static_mode = "none"
            dyn_mode = "none"
            fixed_static: tuple[tuple[float, float, float], ...] = ()
            dyn_min, dyn_max = 1, 2
            if lev == 1:
                static_mode = "fixed"
                fixed_static = ()  # 由 gym_env 使用 builtin_level1_fixed_mixed_3x3
            elif lev == 2:
                static_mode = "random"
            elif lev == 3:
                dyn_mode = "fixed"
            elif lev == 4:
                dyn_mode = "random"
                dyn_min, dyn_max = 1, 3
            elif lev == 5:
                static_mode = "random"
                dyn_mode = "random"
                dyn_min, dyn_max = 1, 2
            _rm = bool(str(getattr(args, "road_map", "") or "").strip())
            env_cfg_kwargs = {
                "control_dt": float(args.control_dt),
                "max_episode_steps": int(args.max_episode_steps),
                "debug_reset": bool(getattr(args, "debug_reset", False)),
                "road_map_yaml": str(getattr(args, "road_map", "") or "").strip(),
                "road_lookahead_n": 5,
                "road_lookahead_ds": 0.8,
                "goal_frame_id": ("world" if _rm else "odom"),
                "done_on_out_of_bounds": (not _rm),
                "done_on_goal": (not _rm),
                "done_on_road_end": (not _rm),
                "done_on_world_goal": _rm,
                "world_goal_success_radius_m": 0.40,
                "spawn_x": float(args.spawn_x),
                "spawn_y": float(args.spawn_y),
                "spawn_z": float(getattr(args, "spawn_z", 0.1)),
                "spawn_yaw": float(args.spawn_yaw),
                "spawn_yaw_towards_goal": bool(getattr(args, "spawn_yaw_towards_goal", False)),
                "goal_x": float(args.goal_x),
                "goal_y": float(args.goal_y),
                "goal_sample_range": goal_sample_range,
                "goal_min_distance": float(args.goal_min_distance),
                "map_x_min": float(args.map_x_min),
                "map_x_max": float(args.map_x_max),
                "map_y_min": float(args.map_y_min),
                "map_y_max": float(args.map_y_max),
                "step_log_csv": None,
                "use_avoid_course_goals": bool(args.avoid_course_goals),
                "avoid_course_curriculum": bool(args.avoid_course_curriculum),
                "curriculum_total_episodes": int(args.curriculum_total_episodes),
                "static_obstacle_mode": str(static_mode),
                "fixed_static_obstacles_xyyaw": tuple(fixed_static),
                "dynamic_obstacle_mode": str(dyn_mode),
                "dynamic_obstacle_count_min": int(dyn_min),
                "dynamic_obstacle_count_max": int(dyn_max),
                "enable_level1_static_obstacles": False,
            }
            sampler = ParallelGazeboSampler(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=int(args.hidden_dim),
                device=str(args.device),
                rollout_steps=int(args.rollout_steps),
                env_cfg_kwargs=env_cfg_kwargs,
                spec_snapshot=(spec.to_snapshot_dict() if spec is not None else None),
                pcfg=pcfg,
            )
            sampler.start()
            sampler.set_weights(trainer.net.state_dict())

        for u in range(start_u, target_u):
            t_u = time.time()
            if sampler is None:
                batch = trainer.collect_rollout()
                mean_r = float(batch.get("mean_step_reward", 0.0))
                mean_abs_r = float(batch.get("mean_abs_step_reward", 0.0))
                episodes_done = int(batch.get("episodes_done", 0))
                mean_ep_len = float(batch.get("mean_ep_len", 0.0))
                tc = int(batch.get("term_collision", 0))
                ts = int(batch.get("term_success", 0))
                toob = int(batch.get("term_oob", 0))
                ttr = int(batch.get("term_trunc", 0))
                toth = int(batch.get("term_other", 0))

                tensor_keys = {"obs", "actions", "logprobs", "advantages", "returns", "values"}
                train_batch = {k: batch[k] for k in tensor_keys if k in batch}
            else:
                batches = sampler.collect(gamma=float(args.gamma), gae_lambda=float(args.gae_lambda))
                # worker 端若失败，会返回 {"error": "..."}；此处直接失败并提示查看对应 worker 日志
                for i, b in enumerate(batches):
                    if isinstance(b, dict) and (b.get("type") == "error" or "error" in b):
                        raise RuntimeError(
                            f"parallel worker {i} failed: {b.get('error')}. "
                            f"See logs in {str(args.parallel_log_dir)!r} (worker_{i}.log)."
                        )
                # concat env batches
                obs = np.concatenate([b["obs"] for b in batches], axis=0)
                actions = np.concatenate([b["actions"] for b in batches], axis=0)
                logp = np.concatenate([b["logprobs"] for b in batches], axis=0)
                adv = np.concatenate([b["advantages"] for b in batches], axis=0)
                ret = np.concatenate([b["returns"] for b in batches], axis=0)
                values = np.concatenate([b["values"] for b in batches], axis=0)
                train_batch = {
                    "obs": obs,
                    "actions": actions,
                    "logprobs": logp,
                    "advantages": adv,
                    "returns": ret,
                    "values": values,
                }
                episodes_done = int(sum(int(b.get("episodes_done", 0)) for b in batches))
                mean_ep_len = float(np.mean([float(b.get("mean_ep_len", 0.0)) for b in batches])) if batches else 0.0
                mean_r = float(np.mean([float(b.get("mean_step_reward", 0.0)) for b in batches])) if batches else 0.0
                mean_abs_r = float(np.mean([float(b.get("mean_abs_step_reward", 0.0)) for b in batches])) if batches else 0.0
                tc = int(sum(int(b.get("term_collision", 0)) for b in batches))
                ts = int(sum(int(b.get("term_success", 0)) for b in batches))
                toob = int(sum(int(b.get("term_oob", 0)) for b in batches))
                ttr = int(sum(int(b.get("term_trunc", 0)) for b in batches))
                toth = int(sum(int(b.get("term_other", 0)) for b in batches))

            stats = trainer.update(train_batch)
            trainer.global_update = int(u + 1)
            print(
                f"[ppo] update {u+1}/{target_u} | mean_step_r={mean_r:.4f} mean| r|={mean_abs_r:.4f} | "
                f"rollout_episodes={episodes_done} mean_ep_len={mean_ep_len:.1f} | "
                f"term{{col={tc} succ={ts} oob={toob} trunc={ttr} other={toth}}} | "
                f"loss_pol={stats['policy_loss']:.5f} loss_val={stats['value_loss']:.5f} "
                f"H={stats['entropy']:.4f} kl={stats['approx_kl']:.5f} | wall_time={time.time()-t_u:.1f}s",
                flush=True,
            )
            if cp_every > 0 and (int(u + 1) % cp_every) == 0:
                trainer.save(args.save)
            if sampler is not None:
                sampler.set_weights(trainer.net.state_dict())
    except BaseException as e:
        save_exc = e
        raise
    finally:
        try:
            if "sampler" in locals() and sampler is not None:
                sampler.close()
        except Exception:
            pass
        try:
            trainer.save(args.save)
            if save_exc is not None:
                print(
                    f"[checkpoint] 抢救保存 -> {args.save} global_update={trainer.global_update} "
                    f"in {time.time()-t0:.1f}s",
                    flush=True,
                )
            else:
                print(f"saved {args.save} in {time.time()-t0:.1f}s", flush=True)
        except Exception as ex:
            print(f"[checkpoint] 写入失败（权重可能未落盘）: {ex}", flush=True)

    if not args.mock and hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
