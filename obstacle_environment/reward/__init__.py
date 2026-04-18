"""
奖励：前进 + 接近目标 + 碰撞惩罚 + 近障惩罚。

典型流程::

    from obstacle_environment.reward import RewardConfig, compute_reward, lidar_min_range

    rb = compute_reward(
        lidar_ranges=scan.ranges,
        linear_x=twist.linear.x,
        goal_distance=obs["goal_distance"],
        prev_goal_distance=prev_gd,
        config=RewardConfig(),
    )
    r = rb.total
"""

from obstacle_environment.reward.reward_computer import (
    RewardBreakdown,
    compute_reward,
    compute_reward_from_observation,
    lidar_front_min_range,
    lidar_min_range,
)
from obstacle_environment.reward.reward_config import RewardConfig

__all__ = [
    "RewardBreakdown",
    "RewardConfig",
    "compute_reward",
    "compute_reward_from_observation",
    "lidar_min_range",
    "lidar_front_min_range",
]
