"""
将 [linear_x, angular_z] 裁剪到 ActionConfig 规定的物理上下界。
"""

from __future__ import annotations

from typing import Union

import numpy as np

from obstacle_environment.action.action_config import ActionConfig

ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]


def clip_action(action: ArrayLike, config: ActionConfig) -> np.ndarray:
    """
    Args:
        action: shape (2,) 或更长（只取前两维）
    Returns:
        float32 ndarray shape (2,)
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 2:
        raise ValueError("clip_action 需要至少 2 维 [linear_x, angular_z]")
    lx = float(np.clip(a[0], config.linear_x_min, config.linear_x_max))
    az = float(np.clip(a[1], config.angular_z_min, config.angular_z_max))
    return np.asarray([lx, az], dtype=np.float32)
