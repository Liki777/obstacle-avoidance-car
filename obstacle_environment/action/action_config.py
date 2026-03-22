"""
动作空间与差速底盘 cmd_vel 约束的统一配置。

与 `geometry_msgs/Twist` 对齐：仅使用 linear.x 与 angular.z（平面差速车）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionConfig:
    """
    Attributes:
        linear_x_min / linear_x_max: 前进速度上下界 (m/s)
        angular_z_min / angular_z_max: 角速度上下界 (rad/s)
        action_dim: 策略输出维度，差速车固定为 2 -> [linear_x, angular_z]
        input_is_normalized: True 时假定策略输出在 [-1, 1]（如 tanh），
            将线性映射到上述物理区间；False 时假定已是物理量，仅做 clip。
    """

    linear_x_min: float = -1.0
    linear_x_max: float = 1.0
    angular_z_min: float = -2.0
    angular_z_max: float = 2.0
    action_dim: int = 2
    input_is_normalized: bool = True

    def __post_init__(self) -> None:
        if int(self.action_dim) != 2:
            raise ValueError("当前差速车接口仅支持 action_dim=2 [linear_x, angular_z]")
        if self.linear_x_min > self.linear_x_max:
            raise ValueError("linear_x_min 不能大于 linear_x_max")
        if self.angular_z_min > self.angular_z_max:
            raise ValueError("angular_z_min 不能大于 angular_z_max")

    def bounds_array(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((lin_low, lin_high), (ang_low, ang_high))"""
        return (
            (float(self.linear_x_min), float(self.linear_x_max)),
            (float(self.angular_z_min), float(self.angular_z_max)),
        )
