from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    # normalize to [-pi, pi]
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def extract_goal_xy(goal_data: Any) -> Optional[np.ndarray]:
    """
    Extract goal position (x,y) from goal_data.
    Supported:
    - dict-like: {"goal_x":..., "goal_y":...} or {"goal_xy": [x,y]}
    - numpy/sequence: [x,y]
    """
    if goal_data is None:
        return None

    if isinstance(goal_data, Mapping):
        if "goal_xy" in goal_data:
            g = np.asarray(goal_data["goal_xy"], dtype=np.float32).reshape(-1)
            return g[:2].copy()
        if "goal_x" in goal_data and "goal_y" in goal_data:
            return np.asarray([goal_data["goal_x"], goal_data["goal_y"]], dtype=np.float32)
        if "x" in goal_data and "y" in goal_data:
            return np.asarray([goal_data["x"], goal_data["y"]], dtype=np.float32)
        return None

    arr = np.asarray(goal_data, dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2].copy()
    return None


def compute_goal_distance_angle(
    *,
    position_xy: Optional[np.ndarray],
    yaw: Optional[float],
    goal_data: Any,
) -> tuple[float, float]:
    """
    Return (goal_distance, goal_angle) where goal_angle is relative to robot heading.
    If pose not available yet, returns (0,0).
    """
    if position_xy is None:
        return 0.0, 0.0
    goal_xy = extract_goal_xy(goal_data)
    if goal_xy is None:
        return 0.0, 0.0

    dx = float(goal_xy[0] - position_xy[0])
    dy = float(goal_xy[1] - position_xy[1])
    dist = float(np.sqrt(dx * dx + dy * dy))

    world_angle = float(np.arctan2(dy, dx))
    if yaw is None:
        return dist, world_angle
    rel = world_angle - float(yaw)
    return dist, _wrap_to_pi(rel)


@dataclass
class GoalProcessor:
    def process(self, odom_data: Any, goal_data: Any) -> tuple[float, float]:
        """
        Returns:
          goal_distance, goal_angle
        """
        # Avoid circular import by local extraction
        from obstacle_environment.observation.odom_processor import extract_pose_xy_yaw

        position_xy, yaw = extract_pose_xy_yaw(odom_data)
        return compute_goal_distance_angle(position_xy=position_xy, yaw=yaw, goal_data=goal_data)

