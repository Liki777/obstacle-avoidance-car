"""
奖励函数权重与阈值（米制，与仿真 / 真车激光一致）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    最终版奖励范式（PPO 稳定收敛模板）：

    R =
      + k_progress * (prev_goal_distance - goal_distance)            # 允许负值（远离目标惩罚）
      + k_safe     * safe_distance(min_lidar)                        # 连续梯度鼓励远离障碍
      - k_risk     * (|v| / min_lidar)                               # 高速贴近障碍强惩罚
      - k_turn     * |w|                                             # 抑制原地转圈
      - k_stop     * 1[v < v_min]                                    # 防止躺平不动
      - k_smooth   * ||a_t - a_{t-1}||_1                             # 防抖

    终止项（由 env 触发）：
      collision -> -collision_penalty
      success   -> +success_reward
      out_of_bounds -> -out_of_bounds_penalty
    """

    # ---- 连续项权重（推荐默认）----
    k_progress: float = 0.5
    k_direction: float = 0.6
    k_velocity_direction: float = 0.6
    k_safe: float = 0.5
    k_risk: float = 0.3
    k_turn: float = 0.05
    k_stop: float = 0.1
    k_smooth: float = 0.01

    # ---- 距离与阈值（m）----
    collision_distance: float = 0.18
    safe_distance: float = 0.55

    # ---- 动作/速度阈值 ----
    v_min: float = 0.05
    risk_eps: float = 0.05

    # ---- 终止奖励 ----
    goal_reached_distance: float = 0.35
    success_reward: float = 100.0
    collision_penalty: float = 100.0
    out_of_bounds_penalty: float = 50.0

    def __post_init__(self) -> None:
        if self.collision_distance >= self.safe_distance:
            raise ValueError("collision_distance 必须小于 safe_distance")
        if self.risk_eps <= 0:
            raise ValueError("risk_eps 必须 > 0")
