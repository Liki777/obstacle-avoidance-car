"""
课程 Level0~5 与 Gazebo world 文件名、默认 8×8 出生位姿（单入口，供 show_map / demo_obstacles / 文档引用）。
"""

from __future__ import annotations

import math

# 与 map_design_document.md、demo_obstacles / train_ppo（--level 1~5 默认 8×8）一致
WORLD_BY_LEVEL: dict[int, str] = {
    0: "level0_arena_8x8.world",
    1: "level1_arena_8x8.world",
    2: "level1_arena_8x8.world",
    3: "level2_arena_8x8.world",
    4: "level2_arena_8x8.world",
    5: "level3_arena_8x8.world",
}

# 默认起点 (1,1)、朝向对角 (7,7)
DEFAULT_ARENA_SPAWN_X = 1.0
DEFAULT_ARENA_SPAWN_Y = 1.0
DEFAULT_ARENA_SPAWN_Z = 0.1
DEFAULT_ARENA_SPAWN_YAW = math.atan2(6.0, 6.0)


def clamp_level(level: int) -> int:
    return max(0, min(5, int(level)))
