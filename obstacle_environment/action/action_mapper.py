"""
策略网络输出 -> 机器人控制量（与 ROS2 `Twist` / `ControlBridge` 对齐）。

典型用法：
- PPO 连续动作经 tanh 落在 [-1,1]，设 `ActionConfig(input_is_normalized=True)`；
- 或网络直接回归物理速度，设 `input_is_normalized=False` 并配合 `clip_action`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Union

import numpy as np

from obstacle_environment.action.action_clipper import clip_action
from obstacle_environment.action.action_config import ActionConfig

ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]


def _map_normalized_to_range(u: float, lo: float, hi: float) -> float:
    """将 [-1, 1] 线性映射到 [lo, hi]。"""
    u = float(np.clip(u, -1.0, 1.0))
    return lo + (u + 1.0) * 0.5 * (hi - lo)


@dataclass
class ActionMapper:
    config: ActionConfig

    def to_linear_angular(self, policy_output: ArrayLike) -> np.ndarray:
        """
        Returns:
            np.ndarray shape (2,) float32: [linear_x, angular_z]
        """
        a = np.asarray(policy_output, dtype=np.float64).reshape(-1)
        if a.size < 2:
            raise ValueError("policy_output 至少需要 2 维")
        if self.config.input_is_normalized:
            lx = _map_normalized_to_range(
                a[0], self.config.linear_x_min, self.config.linear_x_max
            )
            az = _map_normalized_to_range(
                a[1], self.config.angular_z_min, self.config.angular_z_max
            )
            out = np.asarray([lx, az], dtype=np.float32)
        else:
            out = clip_action(a[:2], self.config)
        return out

    def to_cmd_vel_dict(self, policy_output: ArrayLike) -> dict[str, float]:
        """与 `geometry_msgs/Twist` 常用字段名一致，便于填 msg。"""
        v = self.to_linear_angular(policy_output)
        return {"linear_x": float(v[0]), "angular_z": float(v[1])}

    def to_twist_like(self, policy_output: ArrayLike) -> MutableMapping[str, Any]:
        """
        嵌套结构，接近 ROS Twist：
        {"linear": {"x": ...}, "angular": {"z": ...}}
        """
        d = self.to_cmd_vel_dict(policy_output)
        return {"linear": {"x": d["linear_x"], "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": d["angular_z"]}}

    @staticmethod
    def from_cmd_vel_dict(d: Mapping[str, Any]) -> np.ndarray:
        """从扁平 dict 或嵌套 Twist-like dict 还原 [linear_x, angular_z]。"""
        if "linear_x" in d and "angular_z" in d:
            return np.asarray(
                [float(d["linear_x"]), float(d["angular_z"])], dtype=np.float32
            )
        lin = d.get("linear", {})
        ang = d.get("angular", {})
        if isinstance(lin, Mapping) and isinstance(ang, Mapping):
            return np.asarray(
                [float(lin.get("x", 0.0)), float(ang.get("z", 0.0))], dtype=np.float32
            )
        raise ValueError("无法从该 dict 解析 linear_x / angular_z")


def make_action_mapper(config: ActionConfig | None = None) -> ActionMapper:
    return ActionMapper(config=config or ActionConfig())
