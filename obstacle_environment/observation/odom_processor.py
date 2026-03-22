from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # Z axis yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def extract_linear_x(odom_data: Any) -> float:
    """
    车体前向速度 (m/s)，与 base_link / twist.linear.x 一致；用于奖励「前进」项。
    """
    if odom_data is None:
        return 0.0

    if isinstance(odom_data, Mapping):
        if "linear_xy" in odom_data:
            v = np.asarray(odom_data["linear_xy"], dtype=np.float32).reshape(-1)
            if v.size >= 1:
                return float(v[0])
        return float(odom_data.get("linear_x", 0.0))

    if hasattr(odom_data, "twist") and hasattr(odom_data.twist, "twist"):
        lin = odom_data.twist.twist.linear
        return float(lin.x)
    if hasattr(odom_data, "twist") and hasattr(odom_data.twist, "linear"):
        lin = odom_data.twist.linear
        return float(lin.x)

    arr = np.asarray(odom_data, dtype=np.float32)
    if arr.ndim >= 1 and arr.size >= 1:
        return float(arr[0])
    return 0.0


def extract_velocity(odom_data: Any) -> float:
    """
    Extract linear velocity magnitude from odom_data.

    Supported input shapes:
    - dict-like: {"linear_x":..., "linear_y":...} or {"linear_xy": np.array([vx,vy])}
    - ROS2 Odometry msg-like: has pose.pose.position and twist.twist.linear
    - numpy array: [vx, vy] or [vx, vy, yaw,...]
    """
    if odom_data is None:
        return 0.0

    if isinstance(odom_data, Mapping):
        if "linear_xy" in odom_data:
            v = np.asarray(odom_data["linear_xy"], dtype=np.float32).reshape(-1)
            if v.size >= 2:
                return float(np.linalg.norm(v[:2]))
        vx = float(odom_data.get("linear_x", 0.0))
        vy = float(odom_data.get("linear_y", 0.0))
        return float(np.sqrt(vx * vx + vy * vy))

    # ROS msg like
    if hasattr(odom_data, "twist") and hasattr(odom_data.twist, "twist"):
        lin = odom_data.twist.twist.linear
        return float(np.sqrt(float(lin.x) ** 2 + float(lin.y) ** 2))
    if hasattr(odom_data, "twist") and hasattr(odom_data.twist, "linear"):
        lin = odom_data.twist.linear
        return float(np.sqrt(float(lin.x) ** 2 + float(lin.y) ** 2))

    arr = np.asarray(odom_data, dtype=np.float32)
    if arr.ndim >= 1 and arr.size >= 2:
        vx, vy = float(arr[0]), float(arr[1])
        return float(np.sqrt(vx * vx + vy * vy))
    return 0.0


def extract_pose_xy_yaw(odom_data: Any) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Extract (position_xy, yaw) for goal angle computation.

    Returns:
      position_xy: np.ndarray shape (2,) or None
      yaw: float or None
    """
    if odom_data is None:
        return None, None

    if isinstance(odom_data, Mapping):
        if "position_xy" in odom_data:
            p = np.asarray(odom_data["position_xy"], dtype=np.float32).reshape(-1)
            pos = p[:2].copy() if p.size >= 2 else None
        else:
            pos = None
        if "yaw" in odom_data:
            return pos, float(odom_data["yaw"])
        return pos, None

    if hasattr(odom_data, "pose") and hasattr(odom_data.pose, "pose"):
        pos_msg = odom_data.pose.pose.position
        q = odom_data.pose.pose.orientation
        pos = np.asarray([float(pos_msg.x), float(pos_msg.y)], dtype=np.float32)
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        return pos, yaw

    return None, None


@dataclass
class OdomProcessor:
    def process_velocity(self, odom_data: Any) -> float:
        return extract_velocity(odom_data)

    def process_linear_x(self, odom_data: Any) -> float:
        return extract_linear_x(odom_data)

    def process_pose(self, odom_data: Any) -> tuple[Optional[np.ndarray], Optional[float]]:
        return extract_pose_xy_yaw(odom_data)

