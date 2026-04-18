"""
单张避障训练地图：墙体几何 + 区域定义（与 Gazebo world 生成器共用）。
墙体为轴对齐薄盒：厚度 0.05m，高度 0.5m，半厚 hx_thin=0.025m。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

TH = 0.025  # 墙半厚（全厚 0.05m）


@dataclass(frozen=True)
class WallBox:
    """轴对齐盒：中心 (cx,cy)，半宽 hx、半高 hy（全尺寸 2*hx × 2*hy）。"""

    name: str
    cx: float
    cy: float
    hx: float
    hy: float


# 地图总边界（与外墙一致）
MAP_X_MIN = -6.0
MAP_X_MAX = 8.0
MAP_Y_MIN = -4.5
MAP_Y_MAX = 4.5

DEFAULT_START_XY = (-4.5, 0.0)

REGION_TRAIN_A = "train_a"
REGION_TRAIN_B = "train_b"
REGION_TEST = "test"


def build_wall_boxes() -> list[WallBox]:
    """外框 + 开放区/窄道/弯道折线/死胡同/岔路 + 训练测试分隔门洞 + 测试区障碍。"""
    w: list[WallBox] = []

    def hwall(name: str, cx: float, cy: float, length_x: float) -> None:
        """水平墙：沿 x 延伸，厚度 0.05。"""
        w.append(WallBox(name, cx, cy, length_x * 0.5, TH))

    def vwall(name: str, cx: float, cy: float, length_y: float) -> None:
        """竖直墙：沿 y 延伸，厚度 0.05。"""
        w.append(WallBox(name, cx, cy, TH, length_y * 0.5))

    # 外墙
    hwall("outer_bottom", 1.0, -4.5, 14.0)
    hwall("outer_top", 1.0, 4.5, 14.0)
    vwall("outer_left", -6.0, 0.0, 9.0)
    vwall("outer_right", 8.0, 0.0, 9.0)

    # 训练区 A：双竖墙形成 ~1.0m 通道（车可居中）
    vwall("train_a_guide_L", -3.0, 0.0, 3.8)
    vwall("train_a_guide_R", -2.0, 0.0, 3.8)

    # 中央折线墙（弯道半径变化）
    vwall("mid_bend_1", 0.0, -1.2, 3.5)
    hwall("mid_bend_2", 0.8, 1.0, 1.6)
    vwall("mid_bend_3", 1.6, -2.5, 2.0)

    # 窄通道 0.85m：两竖墙中心距 0.85+0.05=0.9
    vwall("narrow_L", 2.075, 0.3, 4.5)
    vwall("narrow_R", 2.925, 0.3, 4.5)

    # 死胡同口袋（约 20% 面积局部）
    vwall("dead_L", 3.5, -3.0, 2.0)
    hwall("dead_back", 4.2, -3.95, 1.4)
    vwall("dead_R", 4.9, -3.0, 2.0)

    # 岔路：横向障碍形成 3 向
    hwall("junction_1", 0.5, 3.0, 3.0)

    # 训练/测试分隔：x=4.35，门洞 y∈[-0.6,0.6]（宽 1.2m）
    # 下段 y∈[-4.5,-0.6]，上段 y∈[0.6,4.5]
    vwall("div_lo", 4.35, -2.55, 3.9)
    vwall("div_hi", 4.35, 2.55, 3.9)

    # 测试区稀疏障碍
    hwall("test_obs1", 6.5, 2.0, 1.0)
    vwall("test_obs2", 7.0, -1.5, 2.0)

    return w


def dist_point_to_wall(px: float, py: float, box: WallBox) -> float:
    dx = abs(px - box.cx)
    dy = abs(py - box.cy)
    qx = max(dx - box.hx, 0.0)
    qy = max(dy - box.hy, 0.0)
    if dx <= box.hx and dy <= box.hy:
        return 0.0
    return float(math.hypot(qx, qy))


def nearest_wall_distance(px: float, py: float, walls: Sequence[WallBox] | None = None) -> float:
    walls = walls or build_wall_boxes()
    return min(dist_point_to_wall(px, py, b) for b in walls)


def is_inside_any_wall(px: float, py: float, walls: Sequence[WallBox] | None = None) -> bool:
    walls = walls or build_wall_boxes()
    for b in walls:
        dx = abs(px - b.cx)
        dy = abs(py - b.cy)
        if dx <= b.hx and dy <= b.hy:
            return True
    return False


def region_of_point(x: float, y: float) -> str:
    if 4.5 <= x <= MAP_X_MAX and MAP_Y_MIN <= y <= MAP_Y_MAX:
        return REGION_TEST
    if -5.0 <= x <= 0.0 and -4.0 <= y <= 4.0:
        return REGION_TRAIN_A
    if 0.0 < x <= 4.0 and -4.0 <= y <= 4.0:
        return REGION_TRAIN_B
    return "other"


def iter_wall_sdf_models(walls: Sequence[WallBox]) -> Iterator[tuple[str, float, float, float, float, float]]:
    """(name, cx, cy, z_center, size_x, size_y) — box 高度 0.5m, z 中心 0.25。"""
    for b in walls:
        sx = 2.0 * b.hx
        sy = 2.0 * b.hy
        yield (b.name, b.cx, b.cy, 0.25, sx, sy)
