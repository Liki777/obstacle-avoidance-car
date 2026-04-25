from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    """
    Observation 配置（前期 PPO 不加入相机，等收敛后再开启 include_camera）。
    """

    lidar_dim: int = 15
    lidar_reduce: str = "min"  # "min" or "mean"

    include_camera: bool = False
    camera_feature_dim: int = 0

    # ---- Road-following features (optional) ----
    include_road: bool = False
    """True 时将道路相关特征拼接到 state 末尾（Level0 道路跟随用）。"""
    road_lookahead_n: int = 5
    """前向路点数量 N；每个路点提供 (x,y) 两维（车体系）。"""

    # state = [lidar, velocity, goal_distance, goal_angle] + 可选 camera 特征
    def state_dim(self) -> int:
        n = int(self.lidar_dim) + 3
        if self.include_camera and int(self.camera_feature_dim) > 0:
            n += int(self.camera_feature_dim)
        if self.include_road:
            n += 2 + 2 * int(self.road_lookahead_n)  # cte, heading_error, lookahead (x,y)*N
        return n

