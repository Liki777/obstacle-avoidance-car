"""
最终版奖励范式（工程可落地 + PPO 稳定收敛）。

R =
  + k_progress * (prev_goal_distance - goal_distance)
  + k_safe     * safe_distance(min_lidar)
  - k_risk     * (|v| / min_lidar)
  - k_turn     * |w|
  - k_stop     * 1[|v| < v_min]
  - k_smooth   * ||a_t - a_{t-1}||_1

终止项（由 env 触发）：
  collision -> -collision_penalty
  success   -> +success_reward
  out_of_bounds -> -out_of_bounds_penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

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
    progress: float
    safe: float
    risk: float
    turn: float
    stop: float
    smooth: float
    terminal: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = {
            "reward_total": float(self.total),
            "reward_progress": float(self.progress),
            "reward_safe": float(self.safe),
            "reward_risk": float(self.risk),
            "reward_turn": float(self.turn),
            "reward_stop": float(self.stop),
            "reward_smooth": float(self.smooth),
            "reward_terminal": float(self.terminal),
        }
        d.update({k: float(v) for k, v in self.components.items()})
        return d


def safe_distance_reward(min_lidar: float, cfg: RewardConfig) -> float:
    """f(d)=clip((d-d_min)/(d_safe-d_min),0,1)"""
    d = float(min_lidar)
    d0 = float(cfg.collision_distance)
    d1 = float(cfg.safe_distance)
    if d1 <= d0:
        return 0.0
    v = (d - d0) / (d1 - d0)
    return float(np.clip(v, 0.0, 1.0))


def progress_reward(goal_distance: float, prev_goal_distance: Optional[float]) -> float:
    if prev_goal_distance is None:
        return 0.0
    return float(prev_goal_distance) - float(goal_distance)


def compute_reward(
    *,
    lidar_ranges: np.ndarray,
    goal_distance: float,
    prev_goal_distance: Optional[float] = None,
    goal_angle: float = 0.0,
    linear_x: float = 0.0,
    angular_z: float = 0.0,
    action: Optional[Sequence[float]] = None,
    prev_action: Optional[Sequence[float]] = None,
    config: Optional[RewardConfig] = None,
    terminal: Optional[str] = None,
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
    v = float(linear_x)
    w = float(angular_z)
    theta = float(goal_angle)

    # 终止覆盖（由 env 传入终止原因）
    if terminal == "collision":
        return RewardBreakdown(
            total=-float(cfg.collision_penalty),
            progress=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            terminal=-float(cfg.collision_penalty),
            components={"lidar_min_m": d_min, "terminal": 1.0},
        )
    if terminal == "success":
        return RewardBreakdown(
            total=float(cfg.success_reward),
            progress=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            terminal=float(cfg.success_reward),
            components={"lidar_min_m": d_min, "terminal": 1.0},
        )
    if terminal == "out_of_bounds":
        return RewardBreakdown(
            total=-float(cfg.out_of_bounds_penalty),
            progress=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            terminal=-float(cfg.out_of_bounds_penalty),
            components={"lidar_min_m": d_min, "terminal": 1.0},
        )

    r_progress = cfg.k_progress * progress_reward(goal_distance, prev_goal_distance)
    # 方向对齐：cos(theta) ∈ [-1, 1]，theta=goal_angle（相对车头方向）
    r_dir = cfg.k_direction * float(np.cos(theta))
    # 动作方向一致性：朝目标方向前进奖励更大，反向前进惩罚
    r_vdir = cfg.k_velocity_direction * (v * float(np.cos(theta)))
    r_safe = cfg.k_safe * safe_distance_reward(d_min, cfg)

    denom = max(float(d_min), float(cfg.risk_eps))
    r_risk = -cfg.k_risk * (abs(v) / denom)

    r_turn = -cfg.k_turn * abs(w)

    r_stop = -cfg.k_stop if abs(v) < float(cfg.v_min) else 0.0

    r_smooth = 0.0
    if action is not None and prev_action is not None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        p = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        n = min(int(a.size), int(p.size))
        if n > 0:
            r_smooth = -cfg.k_smooth * float(np.sum(np.abs(a[:n] - p[:n])))

    total = float(r_progress + r_dir + r_vdir + r_safe + r_risk + r_turn + r_stop + r_smooth)

    return RewardBreakdown(
        total=total,
        progress=float(r_progress),
        safe=float(r_safe),
        risk=float(r_risk),
        turn=float(r_turn),
        stop=float(r_stop),
        smooth=float(r_smooth),
        terminal=0.0,
        components={
            "lidar_min_m": float(d_min),
            "goal_distance": float(goal_distance),
            "goal_angle": float(theta),
            "progress_raw": progress_reward(goal_distance, prev_goal_distance),
            "safe_unit": safe_distance_reward(d_min, cfg),
            "v": v,
            "w": w,
            "cos_theta": float(np.cos(theta)),
            "r_dir": float(r_dir),
            "r_vdir": float(r_vdir),
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
    lx = obs.get("linear_x", obs.get("velocity", 0.0))
    wz = obs.get("angular_z", 0.0)
    gd = float(obs.get("goal_distance", 0.0))
    ga = float(obs.get("goal_angle", 0.0))
    return compute_reward(
        lidar_ranges=lidar_ranges,
        linear_x=float(lx),
        angular_z=float(wz),
        goal_distance=gd,
        goal_angle=ga,
        prev_goal_distance=prev_goal_distance,
        config=config,
        **kwargs,
    )
