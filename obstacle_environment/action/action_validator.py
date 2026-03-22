"""
检查动作是否有限、是否在边界内；用于训练侧与部署侧安全兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from obstacle_environment.action.action_config import ActionConfig

ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    messages: list[str]

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.messages))


def validate_action(action: ArrayLike, config: ActionConfig, *, check_bounds: bool = True) -> ValidationResult:
    """
    - 检查 NaN / Inf
    - 可选检查是否落在 [min, max]（针对已映射到物理空间的向量）
    """
    msgs: list[str] = []
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return ValidationResult(False, ["action 维度不足，至少需要 [linear_x, angular_z]"])

    lx, az = float(a[0]), float(a[1])
    for name, v in (("linear_x", lx), ("angular_z", az)):
        if not np.isfinite(v):
            msgs.append(f"{name} 非有限值: {v}")

    if check_bounds:
        if not (config.linear_x_min <= lx <= config.linear_x_max):
            msgs.append(
                f"linear_x={lx} 超出 [{config.linear_x_min}, {config.linear_x_max}]"
            )
        if not (config.angular_z_min <= az <= config.angular_z_max):
            msgs.append(
                f"angular_z={az} 超出 [{config.angular_z_min}, {config.angular_z_max}]"
            )

    return ValidationResult(len(msgs) == 0, msgs)
