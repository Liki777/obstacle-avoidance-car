"""
Observation builder

唯一对外接口：把 raw 数据（lidar/camera/odom/goal）拼接成 PPO/Actor-Critic 需要的 state 向量。

前期需求：不加入相机数据（include_camera=False），state = [lidar, velocity, goal_distance, goal_angle]
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from obstacle_environment.observation.camera_processor import CameraProcessor
from obstacle_environment.observation.goal_processor import GoalProcessor
from obstacle_environment.observation.lidar_processor import LidarProcessor
from obstacle_environment.observation.odom_processor import OdomProcessor
from obstacle_environment.observation.state_config import ObservationConfig


def build_observation(
    lidar_data: np.ndarray,
    camera_data: Any,
    odom_data: Any,
    goal_data: Any,
    config: Optional[ObservationConfig] = None,
) -> dict[str, Any]:
    cfg = config or ObservationConfig()

    # Lidar -> (lidar_dim,)
    lidar_feat = LidarProcessor(cfg).process(lidar_data)

    # Odom -> 合速度 + 前向速度（奖励「前进」用 linear_x）
    _op = OdomProcessor()
    velocity = float(_op.process_velocity(odom_data))
    linear_x = float(_op.process_linear_x(odom_data))

    # Goal -> (goal_distance, goal_angle)
    goal_distance, goal_angle = GoalProcessor().process(odom_data, goal_data)

    state_parts = [
        lidar_feat,
        np.asarray([velocity, goal_distance, goal_angle], dtype=np.float32),
    ]

    # Camera: 前期不启用，相机特征不加入 state（但保留字段便于后续扩展）
    camera_feat = CameraProcessor(cfg).process(camera_data)
    if cfg.include_camera and camera_feat.size > 0:
        state_parts.append(camera_feat.astype(np.float32, copy=False))

    state_vec = np.concatenate(state_parts, axis=0).astype(np.float32, copy=False)

    return {
        "state": state_vec,  # PPO 输入向量
        "lidar": lidar_feat,
        "velocity": velocity,
        "linear_x": linear_x,
        "goal_distance": float(goal_distance),
        "goal_angle": float(goal_angle),
        "camera": camera_feat,  # 前期为 (0,) 或 zeros
        "odom": odom_data,
        "goal": goal_data,
    }