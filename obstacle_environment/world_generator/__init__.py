"""
地图 / 障碍采样逻辑的可选放置点。

Level1/2/3 的随机静态/动态障碍采样都应放在该包下，保持“纯采样（数据）”与“仿真操作（服务调用）”解耦。
"""

from obstacle_environment.world_generator.static_obstacles import (  # noqa: F401
    StaticObstacleSamplingConfig,
    StaticObstacleSpec,
    sample_static_obstacles,
)
