from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from obstacle_environment.observation.state_config import ObservationConfig


@dataclass
class LidarProcessor:
    config: ObservationConfig

    def _downsample(self, ranges: np.ndarray) -> np.ndarray:
        """
        把 ranges 降到 config.lidar_dim。
        - 默认 min：每个扇区取最小距离，更保守
        """
        target_dim = int(self.config.lidar_dim)
        if ranges.ndim != 1:
            raise ValueError("lidar ranges 必须是一维数组")

        n = int(ranges.shape[0])
        if target_dim <= 0:
            raise ValueError("lidar_dim 必须 > 0")

        if n == target_dim:
            return ranges.astype(np.float32, copy=False)
        if target_dim >= n:
            # 目标维度不小于原始维度，直接返回（避免插值引入误差）
            return ranges.astype(np.float32, copy=True)

        # 使用等宽分箱（按角度均分）
        bins = np.array_split(ranges, target_dim)
        out = np.zeros((target_dim,), dtype=np.float32)
        for i, b in enumerate(bins):
            if b.size == 0:
                out[i] = 0.0
            elif self.config.lidar_reduce == "mean":
                out[i] = float(np.mean(b))
            else:
                out[i] = float(np.min(b))
        return out

    def process(self, lidar_data: np.ndarray) -> np.ndarray:
        """
        输入：
        - lidar_data：LaserScan ranges 转成的一维 numpy（长度可能是 360 或已降到 15）
        输出：
        - lidar_feat：shape=(lidar_dim,)
        """
        ranges = np.asarray(lidar_data, dtype=np.float32)

        # 处理 inf / nan：用最大有限值替换（更合理的“无障碍”）
        finite = ranges[np.isfinite(ranges)]
        if finite.size == 0:
            ranges = np.zeros_like(ranges, dtype=np.float32)
        else:
            rmax = float(np.max(finite))
            ranges = np.nan_to_num(ranges, nan=rmax, posinf=rmax, neginf=0.0).astype(np.float32, copy=False)

        return self._downsample(ranges)


def make_lidar_processor(config: Optional[ObservationConfig] = None) -> LidarProcessor:
    return LidarProcessor(config=config or ObservationConfig())