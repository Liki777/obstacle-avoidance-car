from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StaticObstacleSpec:
    name: str
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class StaticObstacleSamplingConfig:
    """
    Level1：随机静态障碍采样（只生成位置，不做 Gazebo 操作）。
    """

    # 采样边界（world 坐标）
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # 距离约束（m）
    min_dist_to_robot: float = 1.5
    min_dist_to_goal: float = 1.5
    min_dist_between_obstacles: float = 1.0

    # 采样次数上限（避免死循环）
    max_tries_per_obstacle: int = 400

    # 是否随机 yaw
    random_yaw: bool = True


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def sample_static_obstacles(
    *,
    num_obstacles: int,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    cfg: StaticObstacleSamplingConfig,
    rng: np.random.Generator | None = None,
    name_prefix: str = "obstacle",
) -> list[StaticObstacleSpec]:
    """
    返回一组静态障碍物规格（位置 + 名字），不保证“绝对可通行”，但保证基础距离约束。
    若因约束过严导致采样失败，会返回少于 num_obstacles 的数量（尽量多采样）。
    """
    if rng is None:
        rng = np.random.default_rng()

    x0, x1 = float(min(cfg.x_min, cfg.x_max)), float(max(cfg.x_min, cfg.x_max))
    y0, y1 = float(min(cfg.y_min, cfg.y_max)), float(max(cfg.y_min, cfg.y_max))

    out: list[StaticObstacleSpec] = []
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    for i in range(int(num_obstacles)):
        ok = False
        last_xy: tuple[float, float] | None = None
        for _ in range(int(cfg.max_tries_per_obstacle)):
            x = float(rng.uniform(x0, x1))
            y = float(rng.uniform(y0, y1))
            last_xy = (x, y)

            if _dist((x, y), start) < float(cfg.min_dist_to_robot):
                continue
            if _dist((x, y), goal) < float(cfg.min_dist_to_goal):
                continue
            if any(_dist((x, y), (o.x, o.y)) < float(cfg.min_dist_between_obstacles) for o in out):
                continue

            yaw = float(rng.uniform(-math.pi, math.pi)) if cfg.random_yaw else 0.0
            out.append(
                StaticObstacleSpec(
                    name=f"{name_prefix}_{i}",
                    x=x,
                    y=y,
                    z=0.0,
                    yaw=yaw,
                )
            )
            ok = True
            break

        if not ok:
            # 允许返回不满：训练时“有障碍”比“卡死”更重要
            # 记录由上层 logger 完成（该模块保持纯函数）
            if last_xy is not None:
                out.append(
                    StaticObstacleSpec(
                        name=f"{name_prefix}_{i}",
                        x=float(last_xy[0]),
                        y=float(last_xy[1]),
                        z=0.0,
                        yaw=0.0,
                    )
                )
            else:
                break

    return out

