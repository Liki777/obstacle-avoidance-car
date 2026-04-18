"""
场景生命周期（何时删障碍、何时重生）的可选放置点。

约定：该包只做“与 Gazebo/ROS2 服务交互”的管理器，不掺杂采样策略（采样在 world_generator）。
"""

from obstacle_environment.scenario_manager.gazebo_obstacle_manager import (  # noqa: F401
    GazeboObstacleManager,
    GazeboObstacleManagerConfig,
    MixedObstacleSpec,
)
