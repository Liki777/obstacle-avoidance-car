"""
动作侧：策略输出 -> 裁剪 / 校验 -> cmd_vel 语义（差速车）。

与 `obstacle_environment.observation` 配合：观测维度见 `ObservationConfig.state_dim()`，
动作维度见 `ActionConfig.action_dim`（默认 2）。
"""

from obstacle_environment.action.action_clipper import clip_action
from obstacle_environment.action.action_config import ActionConfig
from obstacle_environment.action.action_mapper import ActionMapper, make_action_mapper
from obstacle_environment.action.action_validator import ValidationResult, validate_action

__all__ = [
    "ActionConfig",
    "ActionMapper",
    "ValidationResult",
    "clip_action",
    "make_action_mapper",
    "validate_action",
]
