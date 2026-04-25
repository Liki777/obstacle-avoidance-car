"""
地图 / 障碍采样逻辑的可选放置点。

保持「纯采样（数据）」与「仿真操作（服务调用）」解耦。
"""

from obstacle_environment.world_generator.dynamic_obstacle_presets import (  # noqa: F401
    DynamicObstacleSpec,
    builtin_fixed_dynamic_specs,
    builtin_fixed_static_xyyaw_8x8,
    builtin_level1_fixed_mixed_3x3,
    pose_at_time,
    sample_random_dynamic_specs,
)
from obstacle_environment.world_generator.static_obstacles import (  # noqa: F401
    StaticObstacleSamplingConfig,
    StaticObstacleSpec,
    sample_static_obstacles,
)
