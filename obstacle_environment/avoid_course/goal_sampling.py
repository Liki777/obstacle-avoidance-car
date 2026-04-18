"""
随机终点采样：区域约束 + 离墙距离 + 可选网格 BFS 可达性。
训练时不应在「测试区」采样（由 allow_test_region 控制）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from obstacle_environment.avoid_course.course_layout import (
    DEFAULT_START_XY,
    MAP_X_MAX,
    MAP_X_MIN,
    MAP_Y_MAX,
    MAP_Y_MIN,
    REGION_TEST,
    WallBox,
    build_wall_boxes,
    is_inside_any_wall,
    nearest_wall_distance,
    region_of_point,
)


@dataclass
class GoalSamplingConfig:
    min_distance_from_start: float = 2.0
    min_distance_from_wall: float = 0.5
    map_x_min: float = MAP_X_MIN
    map_x_max: float = MAP_X_MAX
    map_y_min: float = MAP_Y_MIN
    map_y_max: float = MAP_Y_MAX
    allow_test_region: bool = False
    """False：仅在训练区 A/B 采样；True：允许测试区（用于单独评测）。"""
    grid_resolution: float = 0.15
    """BFS 栅格分辨率 (m)。"""
    require_path_exists: bool = True
    prefer_complex_path: bool = True


def _cell_occupied(
    cx: float,
    cy: float,
    walls: list[WallBox],
    robot_radius: float = 0.12,
) -> bool:
    """栅格中心若距任一体墙过近或落在墙内 → 占用。"""
    if is_inside_any_wall(cx, cy, walls):
        return True
    if nearest_wall_distance(cx, cy, walls) < robot_radius + 0.02:
        return True
    return False


def _bfs_reachable(
    start: tuple[float, float],
    goal: tuple[float, float],
    walls: list[WallBox],
    res: float,
) -> bool:
    """简单 4 邻域 BFS，占用由墙膨胀得到。"""
    ix0 = int((start[0] - MAP_X_MIN) / res)
    iy0 = int((start[1] - MAP_Y_MIN) / res)
    ix1 = int((goal[0] - MAP_X_MIN) / res)
    iy1 = int((goal[1] - MAP_Y_MIN) / res)

    nx = int((MAP_X_MAX - MAP_X_MIN) / res) + 1
    ny = int((MAP_Y_MAX - MAP_Y_MIN) / res) + 1

    def idx(ix: int, iy: int) -> int:
        return iy * nx + ix

    occ = [False] * (nx * ny)
    for iy in range(ny):
        for ix in range(nx):
            cx = MAP_X_MIN + (ix + 0.5) * res
            cy = MAP_Y_MIN + (iy + 0.5) * res
            if _cell_occupied(cx, cy, walls):
                occ[idx(ix, iy)] = True

    if not (0 <= ix0 < nx and 0 <= iy0 < ny and 0 <= ix1 < nx and 0 <= iy1 < ny):
        return False
    if occ[idx(ix0, iy0)] or occ[idx(ix1, iy1)]:
        return False

    from collections import deque

    q: deque[tuple[int, int]] = deque()
    seen = [False] * (nx * ny)
    q.append((ix0, iy0))
    seen[idx(ix0, iy0)] = True
    while q:
        ix, iy = q.popleft()
        if ix == ix1 and iy == iy1:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny:
                k = idx(jx, jy)
                if not seen[k] and not occ[k]:
                    seen[k] = True
                    q.append((jx, jy))
    return False


def is_valid_goal(
    goal_xy: tuple[float, float],
    start_xy: tuple[float, float],
    cfg: GoalSamplingConfig,
    walls: Optional[list[WallBox]] = None,
) -> bool:
    x, y = goal_xy
    walls = walls or build_wall_boxes()

    if not (cfg.map_x_min <= x <= cfg.map_x_max and cfg.map_y_min <= y <= cfg.map_y_max):
        return False
    if is_inside_any_wall(x, y, walls):
        return False
    if nearest_wall_distance(x, y, walls) < cfg.min_distance_from_wall:
        return False
    if math.hypot(x - start_xy[0], y - start_xy[1]) < cfg.min_distance_from_start:
        return False

    reg = region_of_point(x, y)
    if not cfg.allow_test_region and reg == REGION_TEST:
        return False

    if cfg.require_path_exists and not _bfs_reachable(start_xy, goal_xy, walls, cfg.grid_resolution):
        return False
    return True


def sample_random_goal(
    start_xy: tuple[float, float],
    cfg: GoalSamplingConfig,
    *,
    walls: Optional[list[WallBox]] = None,
    max_tries: int = 400,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[float, float]]:
    rng = rng or random.Random()
    walls = walls or build_wall_boxes()
    for _ in range(max_tries):
        x = rng.uniform(cfg.map_x_min + 0.3, cfg.map_x_max - 0.3)
        y = rng.uniform(cfg.map_y_min + 0.3, cfg.map_y_max - 0.3)
        if is_valid_goal((x, y), start_xy, cfg, walls):
            return (x, y)
    return None


def curriculum_bin(episode_index: int, total_episodes: int) -> int:
    """0-20% 短距，20-50% 中距，50-80% 长距，80-100% 困难（偏窄区/远距）。"""
    if total_episodes <= 0:
        return 0
    p = episode_index / float(total_episodes)
    if p < 0.2:
        return 0
    if p < 0.5:
        return 1
    if p < 0.8:
        return 2
    return 3


def sample_goal_curriculum(
    episode_index: int,
    total_episodes: int,
    start_xy: tuple[float, float],
    cfg: GoalSamplingConfig,
    *,
    walls: Optional[list[WallBox]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[float, float]]:
    """按难度分箱限制与起点距离范围，再随机合法点。"""
    rng = rng or random.Random()
    walls = walls or build_wall_boxes()
    b = curriculum_bin(episode_index, total_episodes)
    dmin, dmax = {
        0: (2.0, 3.0),
        1: (3.0, 5.0),
        2: (5.0, 8.5),
        3: (4.0, 10.0),
    }[b]

    for _ in range(500):
        x = rng.uniform(cfg.map_x_min + 0.3, cfg.map_x_max - 0.3)
        y = rng.uniform(cfg.map_y_min + 0.3, cfg.map_y_max - 0.3)
        d = math.hypot(x - start_xy[0], y - start_xy[1])
        if not (dmin <= d <= dmax):
            continue
        if b >= 3:
            # 偏好靠窄道/迷宫附近（x>1.5）
            if x < 1.2 and abs(y) < 2.0:
                continue
        if is_valid_goal((x, y), start_xy, cfg, walls):
            return (x, y)
    return sample_random_goal(start_xy, cfg, walls=walls, rng=rng)
