"""
奖励函数权重与阈值（米制，与仿真 / 真车激光一致）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    reward ≈ w_forward * r_fwd
           - w_collision * r_coll
           - w_obstacle * r_near
           + w_goal * r_goal

    各子项内部已做尺度归一或裁剪，权重用于总调参。
    """

    # --- 前进奖励：鼓励沿车体前向 (linear.x) 运动 ---
    w_forward: float = 0.5
    """前进项总权重。"""
    forward_v_ref: float = 1.0
    """参考最大前向速度 (m/s)，用于把 v 归一化到约 [0,1]。"""

    # --- 碰撞惩罚：激光最小距离低于阈值视为碰撞 ---
    w_collision: float = 10.0
    collision_distance: float = 0.18
    """小于等于该距离 (m) 视为碰撞，施加碰撞惩罚。"""

    # --- 靠近障碍惩罚：在 (collision_distance, obstacle_safe_distance] 区间内连续惩罚 ---
    w_obstacle: float = 1.0
    obstacle_safe_distance: float = 0.55
    """大于该距离认为不受「近障」惩罚；需大于 collision_distance。"""

    # --- 接近目标奖励 ---
    w_goal: float = 2.0
    goal_progress_mode: str = "delta"
    """
    - "delta": 需要上一时刻 goal_distance，奖励为 (prev - curr) 的正值部分（密集引导）
    - "potential": 使用 -goal_distance / goal_distance_ref 的势函数（单步即可）
    """
    goal_distance_ref: float = 10.0
    """potential 模式下的尺度 (m)。"""

    goal_reached_distance: float = 0.35
    """到达判定距离 (m)，可用于外部 done；本模块可给额外成功奖励。"""
    w_goal_success: float = 50.0
    """首次进入 goal_reached_distance 内的一次性奖励（可选，由 compute 参数开启）。"""

    def __post_init__(self) -> None:
        if self.collision_distance >= self.obstacle_safe_distance:
            raise ValueError("collision_distance 必须小于 obstacle_safe_distance")
        if self.goal_progress_mode not in ("delta", "potential"):
            raise ValueError('goal_progress_mode 必须是 "delta" 或 "potential"')
