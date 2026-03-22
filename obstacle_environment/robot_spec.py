"""
将「观测配置 + 动作配置」绑在一起，便于 PPO / 部署脚本统一引用维度与边界。

示例::

    from obstacle_environment import RobotTaskSpec, build_observation
    from obstacle_environment.action import ActionMapper

    spec = RobotTaskSpec.preset_diff_drive()
    assert spec.state_dim == spec.observation_config.state_dim()
    obs = build_observation(lidar, cam, odom, goal, spec.observation_config)
    mapper = ActionMapper(spec.action_config)
    cmd = mapper.to_cmd_vel_dict(policy_logits_after_tanh)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from obstacle_environment.action.action_config import ActionConfig
from obstacle_environment.observation.state_config import ObservationConfig
from obstacle_environment.reward.reward_config import RewardConfig


@dataclass(frozen=True)
class RobotTaskSpec:
    """单机仿真/实车任务的一条龙规格（观测 + 动作 + 奖励）。"""

    observation_config: ObservationConfig
    action_config: ActionConfig
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    @property
    def state_dim(self) -> int:
        return self.observation_config.state_dim()

    @property
    def action_dim(self) -> int:
        return int(self.action_config.action_dim)

    @classmethod
    def preset_diff_drive(
        cls,
        *,
        lidar_dim: int = 15,
        include_camera: bool = False,
        camera_feature_dim: int = 0,
        input_is_normalized: bool = True,
        reward_config: RewardConfig | None = None,
    ) -> RobotTaskSpec:
        """默认与 `observation_builder` / 差速 Gazebo 小车常见设置一致。"""
        return cls(
            observation_config=ObservationConfig(
                lidar_dim=lidar_dim,
                lidar_reduce="min",
                include_camera=include_camera,
                camera_feature_dim=camera_feature_dim,
            ),
            action_config=ActionConfig(input_is_normalized=input_is_normalized),
            reward_config=reward_config or RewardConfig(),
        )
