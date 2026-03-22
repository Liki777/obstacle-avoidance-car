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

    # state = [lidar, velocity, goal_distance, goal_angle] + 可选 camera 特征
    def state_dim(self) -> int:
        n = int(self.lidar_dim) + 3
        if self.include_camera and int(self.camera_feature_dim) > 0:
            n += int(self.camera_feature_dim)
        return n

