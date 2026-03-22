"""
obstacle_environment
====================

**观测**：`build_observation` + `ObservationConfig`（激光下采样、速度、目标相对角/距，可选相机占位特征）

**动作**：`ActionConfig`、`ActionMapper`、`clip_action`、`validate_action`（差速 cmd_vel 语义）

**整合**：`RobotTaskSpec` 统一 state_dim / action_dim 与预设。

**奖励**：`RewardConfig`、`compute_reward`（前进 / 碰撞 / 近障 / 接近目标）。

**Gazebo 环境**（惰性导入，需 ``rclpy``）：``from obstacle_environment import RlCarGazeboEnv, GazeboEnvConfig``
或 ``from obstacle_environment.gym_env import RlCarGazeboEnv``。
"""

from obstacle_environment.action import (
    ActionConfig,
    ActionMapper,
    ValidationResult,
    clip_action,
    make_action_mapper,
    validate_action,
)
from obstacle_environment.observation import ObservationConfig, build_observation
from obstacle_environment.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    compute_reward_from_observation,
    lidar_min_range,
)
from obstacle_environment.robot_spec import RobotTaskSpec

__all__ = [
    "ActionConfig",
    "ActionMapper",
    "ObservationConfig",
    "RewardBreakdown",
    "RewardConfig",
    "RobotTaskSpec",
    "ValidationResult",
    "build_observation",
    "clip_action",
    "compute_reward",
    "compute_reward_from_observation",
    "lidar_min_range",
    "make_action_mapper",
    "validate_action",
    # 惰性导出（需 rclpy）：
    "GazeboEnvConfig",
    "RlCarGazeboEnv",
]


def __getattr__(name: str):
    if name == "GazeboEnvConfig":
        from obstacle_environment.gym_env import GazeboEnvConfig

        return GazeboEnvConfig
    if name == "RlCarGazeboEnv":
        from obstacle_environment.gym_env import RlCarGazeboEnv

        return RlCarGazeboEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
