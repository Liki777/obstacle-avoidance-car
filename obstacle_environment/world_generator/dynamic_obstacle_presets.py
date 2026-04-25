"""
动态障碍：解析运动轨迹（用于 Gazebo 每步 SetEntityState 更新位姿）。

与 ``GazeboEnvConfig.dynamic_obstacle_mode`` 配合：
- ``fixed``：使用 ``builtin_fixed_dynamic_specs()``（可复现的正弦/圆周）
- ``random``：``sample_random_dynamic_specs`` 在每次 episode reset 采样
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal


Pattern = Literal["sin_x", "sin_y", "circle"]


@dataclass(frozen=True)
class DynamicObstacleSpec:
    """单个动态盒子的运动参数（世界系）。"""

    name: str
    pattern: Pattern
    x0: float
    y0: float
    amplitude: float
    omega: float
    phase: float
    yaw: float


def pose_at_time(sp: DynamicObstacleSpec, t: float) -> tuple[float, float, float]:
    """返回 (x, y, yaw)。"""
    w = float(sp.omega) * float(t) + float(sp.phase)
    a = float(sp.amplitude)
    if sp.pattern == "sin_x":
        x = float(sp.x0) + a * math.sin(w)
        y = float(sp.y0)
    elif sp.pattern == "sin_y":
        x = float(sp.x0)
        y = float(sp.y0) + a * math.sin(w)
    else:  # circle
        x = float(sp.x0) + a * math.cos(w)
        y = float(sp.y0) + a * math.sin(w)
    return x, y, float(sp.yaw)


def builtin_fixed_dynamic_specs(*, name_prefix: str = "train_dyn") -> list[DynamicObstacleSpec]:
    """
    Level3：固定、可复现的少量动态障碍（正弦 / 圆周混合）。
    坐标针对 8×8 场地中心区域，避免贴墙。
    """
    p = name_prefix
    return [
        DynamicObstacleSpec(
            name=f"{p}_0",
            pattern="sin_x",
            x0=4.0,
            y0=4.0,
            amplitude=1.15,
            omega=0.65,
            phase=0.0,
            yaw=0.0,
        ),
        DynamicObstacleSpec(
            name=f"{p}_1",
            pattern="sin_y",
            x0=5.5,
            y0=4.0,
            amplitude=0.95,
            omega=0.85,
            phase=1.2,
            yaw=0.0,
        ),
    ]


def builtin_fixed_static_xyyaw_8x8() -> tuple[tuple[float, float, float], ...]:
    """遗留：三盒固定布局。Level1 默认已改用 ``builtin_level1_fixed_mixed_3x3``。"""
    return (
        (3.5, 4.0, 0.7853981633974483),
        (5.2, 2.6, 0.0),
        (2.1, 5.4, 1.5707963267948966),
    )


def builtin_level1_fixed_mixed_3x3(*, name_prefix: str = "train_l1") -> list[Any]:
    """
    Level1：场地中部 3×3 共 9 个固定障碍（与 ``spawn_mixed_static`` 一致）。

    **由上往下**三行（y 从大到小）、左到右为 x 增：
    - 第 1 行：圆柱 | 方柱 | 扁圆
    - 第 2 行：方柱 | 圆柱 | 扁圆
    - 第 3 行：方柱 | 扁圆 | 圆柱

    栅格中心 x∈{2.5,4.0,5.5}，y∈{5.5,4.0,2.5}（上行靠场地北侧/目标侧）；相邻格中心距 **1.5 m**。

    **通路宽度（与碰撞几何一致，粗算相邻两格沿 x 或 y 轴线方向）**：
    车体底盘碰撞盒约 **0.335×0.265 m**（``rl_car_description/robot.urdf`` 中 chassis）。
    方柱 footprint 半边长 **0.17 m**（sx=sy=0.34）；竖圆柱半径 **0.19 m**；扁圆柱半径 **0.32 m**。
    相邻中心距 1.5 m 时，两障碍之间净距约为 ``1.5 - r_a - r_b``（圆-圆）或 ``1.5 - 0.17 - 0.17``（方-方邻接）等，
    最紧组合之一为扁圆邻竖圆柱：**1.5 - 0.32 - 0.19 ≈ 0.99 m**，仍远大于车体等效宽度（≈0.34 m），**几何上可从格间穿过**，不要求从障碍「正中心」穿过。
    """
    from obstacle_environment.scenario_manager.gazebo_obstacle_manager import MixedObstacleSpec

    xs = (2.5, 4.0, 5.5)
    ys = (5.5, 4.0, 2.5)
    # 与用户需求一致的 3×3 类型布局（key: cyl=竖圆柱, cube=方柱, flat=扁圆柱）
    layout: tuple[tuple[str, str, str], ...] = (
        ("cyl", "cube", "flat"),
        ("cube", "cyl", "flat"),
        ("cube", "flat", "cyl"),
    )
    out: list[MixedObstacleSpec] = []
    for iy in range(3):
        for ix in range(3):
            idx = iy * 3 + ix
            x = float(xs[ix])
            y = float(ys[iy])
            cell = layout[iy][ix]
            yaw = float((ix * 0.31 + iy * 0.27) % 1.2)
            name = f"{name_prefix}_{idx:02d}"
            if cell == "flat":
                # 扁圆：SDF 圆柱轴向为 Z，「长度」sy 即总高度。必须 ≥ 车体激光扫描高度（URDF 激光约 z≈0.21m），
                # 否则水平扫描平面从障碍物顶上方掠过，雷达测不到体素，dmin/奖励与 Gazebo 碰撞不一致。
                out.append(
                    MixedObstacleSpec(
                        name=name,
                        kind="cylinder",
                        x=x,
                        y=y,
                        yaw=yaw,
                        sx=0.32,
                        sy=0.38,
                        sz=0.0,
                    )
                )
            elif cell == "cube":
                out.append(
                    MixedObstacleSpec(
                        name=name,
                        kind="cube",
                        x=x,
                        y=y,
                        yaw=yaw,
                        sx=0.34,
                        sy=0.34,
                        sz=0.52,
                    )
                )
            else:  # cyl
                out.append(
                    MixedObstacleSpec(
                        name=name,
                        kind="cylinder",
                        x=x,
                        y=y,
                        yaw=yaw,
                        sx=0.19,
                        sy=0.52,
                        sz=0.0,
                    )
                )
    return out


def sample_random_dynamic_specs(
    rng: random.Random,
    *,
    count: int,
    map_x_min: float,
    map_x_max: float,
    map_y_min: float,
    map_y_max: float,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    name_prefix: str = "train_dyn",
) -> list[DynamicObstacleSpec]:
    """Level4/5：随机动态障碍（模式与参数在 reset 时一次性采样）。"""
    out: list[DynamicObstacleSpec] = []
    margin = 0.55
    gx0, gx1 = float(map_x_min) + margin, float(map_x_max) - margin
    gy0, gy1 = float(map_y_min) + margin, float(map_y_max) - margin
    patterns: list[Pattern] = ["sin_x", "sin_y", "circle"]
    for i in range(int(count)):
        for _ in range(80):
            x0 = rng.uniform(gx0, gx1)
            y0 = rng.uniform(gy0, gy1)
            if math.hypot(x0 - start_xy[0], y0 - start_xy[1]) < 1.15:
                continue
            if math.hypot(x0 - goal_xy[0], y0 - goal_xy[1]) < 1.15:
                continue
            pat = rng.choice(patterns)
            amp = rng.uniform(0.45, 1.25)
            if pat == "circle":
                amp = rng.uniform(0.35, 0.85)
            omega = rng.uniform(0.45, 1.15)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            if pat != "circle":
                if x0 + amp > gx1 or x0 - amp < gx0 or y0 + amp > gy1 or y0 - amp < gy0:
                    continue
            else:
                if x0 + amp > gx1 or x0 - amp < gx0 or y0 + amp > gy1 or y0 - amp < gy0:
                    continue
            out.append(
                DynamicObstacleSpec(
                    name=f"{name_prefix}_{i}",
                    pattern=pat,
                    x0=x0,
                    y0=y0,
                    amplitude=amp,
                    omega=omega,
                    phase=phase,
                    yaw=rng.uniform(-0.2, 0.2),
                )
            )
            break
    return out
