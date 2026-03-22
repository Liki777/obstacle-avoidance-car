"""
根据激光最小距、前向速度、目标距离（及可选上一时刻距离）计算标量奖励与分项。

设计对应关系：
  + 前进奖励
  - 碰撞惩罚
  - 靠近障碍惩罚
  + 接近目标奖励
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from obstacle_environment.reward.reward_config import RewardConfig


def lidar_min_range(lidar_ranges: np.ndarray) -> float:
    """一维激光距离数组的最小有限值 (m)；无有效数据时返回大数。"""
    r = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return float(1e3)
    return float(np.min(finite))


@dataclass
class RewardBreakdown:
    """分项与总和，便于 TensorBoard / CSV 记录。"""

    total: float
    forward: float
    collision: float
    obstacle: float
    goal: float
    success_bonus: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = {
            "reward_total": float(self.total),
            "reward_forward": float(self.forward),
            "reward_collision": float(self.collision),
            "reward_obstacle": float(self.obstacle),
            "reward_goal": float(self.goal),
            "reward_success_bonus": float(self.success_bonus),
        }
        d.update({k: float(v) for k, v in self.components.items()})
        return d


def _forward_reward(linear_x: float, cfg: RewardConfig) -> float:
    v = max(0.0, float(linear_x))
    ref = max(cfg.forward_v_ref, 1e-6)
    return v / ref  # 约 [0,1]


def _collision_penalty(lidar_min: float, cfg: RewardConfig) -> float:
    """碰撞为正值惩罚量（后续用减号）。"""
    if lidar_min <= cfg.collision_distance:
        return 1.0
    return 0.0


def _obstacle_penalty(lidar_min: float, cfg: RewardConfig) -> float:
    """
    非碰撞但过近时的惩罚量 ∈ [0,1]，在 safe 边界为 0，在 collision 边界为 1。
    """
    d0 = cfg.collision_distance
    d1 = cfg.obstacle_safe_distance
    if lidar_min <= d0:
        return 0.0  # 交给碰撞项
    if lidar_min >= d1:
        return 0.0
    return (d1 - lidar_min) / (d1 - d0)


def _goal_reward(
    goal_distance: float,
    prev_goal_distance: Optional[float],
    cfg: RewardConfig,
) -> float:
    g = float(max(0.0, goal_distance))
    if cfg.goal_progress_mode == "potential":
        ref = max(cfg.goal_distance_ref, 1e-6)
        return -g / ref  # 越近越大（负得少），范围约 [-1,0]

    # delta
    if prev_goal_distance is None:
        return 0.0
    prev = float(prev_goal_distance)
    # 更接近目标为正
    return max(0.0, prev - g)


def compute_reward(
    *,
    lidar_ranges: np.ndarray,
    linear_x: float,
    goal_distance: float,
    prev_goal_distance: Optional[float] = None,
    config: Optional[RewardConfig] = None,
    give_success_bonus: bool = False,
    was_goal_reached_before: bool = False,
) -> RewardBreakdown:
    """
    Args:
        lidar_ranges: 原始或下采样前的一维距离数组
        linear_x: 车体前向速度 (m/s)，与 ROS base_link 前向一致时取 twist.linear.x
        goal_distance: 当前到目标欧氏距离
        prev_goal_distance: 上一时刻距离，用于 delta 型接近目标奖励
        give_success_bonus: 是否发放一次性到达奖励
        was_goal_reached_before: 若已为 True，则不再发 success_bonus
    """
    cfg = config or RewardConfig()
    d_min = lidar_min_range(lidar_ranges)

    r_fwd = _forward_reward(linear_x, cfg)
    p_coll = _collision_penalty(d_min, cfg)
    p_obs = _obstacle_penalty(d_min, cfg)
    r_goal_raw = _goal_reward(goal_distance, prev_goal_distance, cfg)

    forward_term = cfg.w_forward * r_fwd
    collision_term = cfg.w_collision * p_coll
    obstacle_term = cfg.w_obstacle * p_obs
    goal_term = cfg.w_goal * r_goal_raw

    success_bonus = 0.0
    if give_success_bonus and (not was_goal_reached_before):
        if float(goal_distance) <= cfg.goal_reached_distance:
            success_bonus = cfg.w_goal_success

    total = (
        forward_term
        - collision_term
        - obstacle_term
        + goal_term
        + success_bonus
    )

    return RewardBreakdown(
        total=total,
        forward=forward_term,
        collision=-collision_term,
        obstacle=-obstacle_term,
        goal=goal_term,
        success_bonus=success_bonus,
        components={
            "lidar_min_m": d_min,
            "r_fwd_normalized": r_fwd,
            "p_collision_unit": p_coll,
            "p_obstacle_unit": p_obs,
            "r_goal_raw": r_goal_raw,
        },
    )


def compute_reward_from_observation(
    obs: Mapping[str, Any],
    *,
    lidar_ranges: np.ndarray,
    prev_goal_distance: Optional[float] = None,
    config: Optional[RewardConfig] = None,
    **kwargs: Any,
) -> RewardBreakdown:
    """
    与 `build_observation` 返回的 dict 配合使用。

    - 优先使用 ``obs["linear_x"]``（车体前向速度）作为前进奖励；
      若无则回退到 ``velocity``（合速度模长）。
    - 激光最小距需传入原始 ``lidar_ranges``（一维，与 LaserScan.ranges 一致）。
    """
    lx = obs.get("linear_x")
    if lx is None:
        lx = obs.get("velocity", 0.0)
    gd = float(obs.get("goal_distance", 0.0))
    return compute_reward(
        lidar_ranges=lidar_ranges,
        linear_x=float(lx),
        goal_distance=gd,
        prev_goal_distance=prev_goal_distance,
        config=config,
        **kwargs,
    )
