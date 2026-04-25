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
    out_of_road_penalty: float = 60.0

    # ---- 道路跟随（可选）：若启用，将用 Δs 替代/补充目标 progress ----
    k_road_progress: float = 0.0
    k_road_cte: float = 0.0
    k_road_heading: float = 0.0
    # ---- 车道标线惩罚（可选，需 env 传入 robot_xy 与道路几何参数）----
    k_road_yellow_line: float = 0.0
    """压双黄实线区域（靠近道路中心 x_center）惩罚权重。"""
    k_road_white_line: float = 0.0
    """压白边线区域惩罚权重（应明显小于黄线，允许必要时短暂跨线避障）。"""
    k_road_lane_divider: float = 0.0
    """压车道分隔线（同向两车道之间）惩罚权重（介于黄/白之间）。"""
    road_x_center: float = 4.0
    road_yellow_line_x_a: float = 3.97
    road_yellow_line_x_b: float = 4.03
    road_yellow_half_width_m: float = 0.02
    road_white_half_width_m: float = 0.03
    road_lane_divider_half_width_m: float = 0.015
    road_white_edge_left_x: float = 2.57
    road_white_edge_right_x: float = 5.43
    road_lane_divider_left_x: float = 3.25
    road_lane_divider_right_x: float = 4.75
    # ---- 倒车惩罚（用车体前向速度 linear_x）----
    k_reverse: float = 0.0
    reverse_v_eps: float = 0.02

    def __post_init__(self) -> None:
        if self.collision_distance >= self.safe_distance:
            raise ValueError("collision_distance 必须小于 safe_distance")
        if self.risk_eps <= 0:
            raise ValueError("risk_eps 必须 > 0")

    @staticmethod
    def stage2_level1_corridor() -> "RewardConfig":
        """
        Level1 九宫格混合静障：在 ``stage2_simple_avoid`` 基础上略提高进度/朝向权重、略降低转弯惩罚，
        便于在 1.5 m 格距走廊中连续绕行，同时保留前向扇区安全项。
        """
        return RewardConfig(
            k_progress=0.32,
            k_direction=0.38,
            k_velocity_direction=0.42,
            k_behind_goal=0.45,
            k_safe=0.62,
            k_risk=0.38,
            k_front_safe=0.22,
            k_front_risk=0.34,
            k_turn=0.55,
            k_stop=0.08,
            k_smooth=0.015,
            k_front_close_penalty=1.55,
            front_close_penalty_margin_m=0.72,
            front_sector_half_width_rad=0.52,
            collision_distance=0.19,
            safe_distance=0.56,
            success_reward=85.0,
            collision_penalty=85.0,
        )

    @staticmethod
    def stage0_road_follow() -> "RewardConfig":
        """Level0 道路行驶：主要看 Δs 前进，不强拉“中心线”，重点是不轻易跨黄线/车道分隔白线。"""
        return RewardConfig(
            # disable goal-centric shaping (use road instead)
            k_progress=0.0,
            k_direction=0.0,
            k_velocity_direction=0.0,
            k_behind_goal=0.0,
            k_lateral_detour=0.0,
            k_front_safe=0.0,
            k_front_risk=0.0,
            # road terms
            k_road_progress=3.2,
            # 不用 cte/heading 去“拉线”，否则会与标线惩罚打架（尤其双黄线在道路中间）
            k_road_cte=0.05,
            k_road_heading=0.05,
            # lane markings (soft penalties)
            # 这里的 half_width 不是“视觉线宽”，而是“禁止轻易跨线”的缓冲带宽度（越过越深罚越大）
            k_road_yellow_line=3.0,
            k_road_white_line=0.25,
            k_road_lane_divider=2.0,
            # discourage reversing in normal driving
            k_reverse=1.2,
            reverse_v_eps=0.02,
            # gentle control regularization
            k_turn=0.05,
            k_stop=0.25,
            k_smooth=0.01,
            # thresholds
            collision_distance=0.16,
            safe_distance=0.55,
            goal_reached_distance=0.35,
            # terminal
            success_reward=60.0,
            collision_penalty=70.0,
            out_of_bounds_penalty=40.0,
            out_of_road_penalty=70.0,
            # buffers (m)
            road_yellow_half_width_m=0.18,
            road_white_half_width_m=0.10,
            road_lane_divider_half_width_m=0.14,
        )

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
            collision_distance=0.2,
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
