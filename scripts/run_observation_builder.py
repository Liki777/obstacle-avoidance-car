"""
仅用于离线快速检查 observation_builder 的拼接逻辑（不连接 ROS）。

真实仿真/真机请使用：
  scripts/collect_observations.py
  scripts/run_launch_and_collect_obs.sh

下面用固定数值（非随机）演示 shape 与键名。
"""
from __future__ import annotations

import numpy as np

from obstacle_environment.observation import ObservationConfig, build_observation

if __name__ == "__main__":
    lidar = np.linspace(0.5, 5.0, 360, dtype=np.float32)
    camera = np.zeros((64, 64, 3), dtype=np.uint8)
    odom = {
        "linear_xy": np.array([0.2, 0.0], dtype=np.float32),
        "position_xy": np.array([1.0, 2.0], dtype=np.float32),
        "yaw": 0.0,
    }
    goal = {"goal_xy": np.array([2.0, 2.0], dtype=np.float32)}
    cfg = ObservationConfig(lidar_dim=15, lidar_reduce="min", include_camera=False)

    obs = build_observation(lidar, camera, odom, goal, config=cfg)
    print("state shape:", obs["state"].shape)
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}")
        else:
            print(f"{k}: {type(v).__name__}")
