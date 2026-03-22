from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from obstacle_environment.observation.state_config import ObservationConfig


@dataclass
class CameraProcessor:
    config: ObservationConfig

    def process(self, camera_data: Any) -> np.ndarray:
        """
        前期不使用相机：返回空特征向量，供后续扩展。
        """
        if self.config.include_camera:
            # 这里先不实现 CNN 特征提取（需要你指定编码/尺寸/网络）
            # 返回占位，避免训练时 shape 不一致。
            return np.zeros((int(self.config.camera_feature_dim),), dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)


def make_camera_processor(config: ObservationConfig) -> CameraProcessor:
    return CameraProcessor(config=config)

