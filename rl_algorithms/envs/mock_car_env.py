"""玩具环境：与 preset_diff_drive 维度一致，用于无 ROS 时验证 PPO。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MockCarEnv:
    state_dim: int = 18
    action_dim: int = 2
    max_steps: int = 256

    def __post_init__(self) -> None:
        self._s: Optional[np.ndarray] = None
        self._t = 0

    def reset(self) -> np.ndarray:
        self._s = (np.random.randn(self.state_dim).astype(np.float32) * 0.2).astype(np.float32)
        self._t = 0
        return self._s.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        assert self._s is not None
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < self.action_dim:
            raise ValueError(f"action 至少 {self.action_dim} 维")
        a = np.clip(a[: self.action_dim], -1.0, 1.0)

        noise = np.random.randn(self.state_dim).astype(np.float32) * 0.02
        delta = np.concatenate([a, np.zeros(self.state_dim - 2, dtype=np.float32)])
        self._s = self._s + 0.08 * delta + noise
        self._s = np.clip(self._s, -5.0, 5.0).astype(np.float32)

        n = float(np.linalg.norm(self._s))
        reward = -0.01 * n + 0.1 * float(np.exp(-n))
        self._t += 1
        done = self._t >= self.max_steps or n < 0.15
        if n < 0.15:
            reward += 5.0
        return self._s.copy(), reward, done, {"mock_norm": n}
