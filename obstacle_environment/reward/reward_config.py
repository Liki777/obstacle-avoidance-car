"""
奖励函数权重与阈值（米制，与仿真 / 真车激光一致）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    R 主要项：
      + k_progress * progress_raw
          - 默认：位移在世界系下向「当前位置→目标」单位向量上的投影（抑制绕圈刷线速度）
          - 可选回退：(prev_goal_distance - goal_distance)
      + k_direction * cos(goal_angle)（默认 k_direction=1 即纯 cos）- k_behind_goal * max(0, |goal_angle| - pi/2)
      - k_front_close_penalty * max(0, margin - lidar_front_min)（前向过近额外罚）
      + k_lateral_detour（|world_y| 超过 lateral_detour_thresh_m 时，鼓励绕开中线障碍）
      + k_velocity_direction * v * cos(goal_angle)
      + k_safe * safe_distance(min_lidar) - k_risk * (|v|/min_lidar) - ...
    """

    # ---- 连续项权重（推荐默认）----
    k_progress: float = 0.4
    k_direction: float = 1.0
    k_velocity_direction: float = 0.45
    k_safe: float = 0.1
    k_risk: float = 0.3
    k_turn: float = 0.3
    k_stop: float = 0.3
    k_smooth: float = 0.1

    # ---- 进度：位移投影（推荐 True，避免「高速前进+转弯」绕圈骗距离差）----
    use_displacement_progress: bool = True
    """True：用上一时刻与当前位姿的位移在「指向目标」方向上的投影作为 progress_raw。"""
    progress_heading_gate_rad: float = math.pi / 4.0
    """|goal_angle| 超过此角时，将 progress 项乘以 progress_heading_gate_scale（先对准再走）。"""
    progress_heading_gate_scale: float = 0.1
    k_behind_goal: float = 0.35
    """|goal_angle| > pi/2 时额外惩罚：减去 k_behind_goal * (|angle| - pi/2)。"""
    # ---- 前向过近硬罚（与 lidar_front_min 配合，提前减速仍贴前时施压）----
    k_front_close_penalty: float = 2.0
    front_close_penalty_margin_m: float = 0.8
    """d_front < margin 时加罚 -k_front_close_penalty * (margin - d_front)。"""
    # ---- 侧向绕障（世界系 pos_y，走廊中线障碍场景）----
    k_lateral_detour: float = 0.2
    lateral_detour_thresh_m: float = 0.5
    """|y| > thresh 时 +k_lateral_detour。"""
    # ---- 前向扇区雷达（相对车头 ±front_sector_half_width_rad）----
    k_front_safe: float = 0.0
    k_front_risk: float = 0.0
    front_sector_half_width_rad: float = 0.52

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

    @staticmethod
    def stage2_simple_avoid() -> "RewardConfig":
        """阶段2：简单避障——强化雷达安全/风险与前向扇区。"""
        return RewardConfig(
            k_progress=0.25,
            k_direction=0.25,
            k_velocity_direction=0.4,
            k_behind_goal=0.6,
            k_safe=0.75,
            k_risk=0.45,
            k_front_safe=0.28,
            k_front_risk=0.4,
            k_turn=0.9,
            k_stop=0.08,
            k_smooth=0.01,
            front_sector_half_width_rad=0.52,
            collision_distance=0.35,
            safe_distance=0.58,
            success_reward=80.0,
            collision_penalty=80.0,
        )

    @staticmethod
    def stage3_hard_avoid() -> "RewardConfig":
        """阶段3：复杂避障——更强雷达惩罚与更紧安全距离。"""
        return RewardConfig(
            k_progress=0.4,
            k_direction=1.0,
            k_velocity_direction=0.38,
            k_behind_goal=0.5,
            k_safe=0.95,
            k_risk=0.65,
            k_front_safe=0.35,
            k_front_risk=0.55,
            k_turn=0.05,
            k_stop=0.1,
            k_smooth=0.012,
            front_sector_half_width_rad=0.65,
            collision_distance=0.20,
            safe_distance=0.62,
            risk_eps=0.06,
            success_reward=100.0,
            collision_penalty=120.0,
        )
