#!/usr/bin/env python3
"""
通用地图查看入口：启动 Gazebo 加载指定 world。

两种方式（二选一）：

1. **按课程阶段**（推荐，与 ``run.py train --level`` / ``demo_obstacles`` 一致）::

    python3 show_map.py --level 1

2. **直接指定 world 名**::

    python3 show_map.py level1_map
    python3 show_map.py level1_arena_8x8.world

使用 ``--level`` 时会自动选用 ``curriculum_maps.WORLD_BY_LEVEL`` 中的 world，
并默认出生位姿为 8×8 场地 (1,1)、车头朝向对角目标；可用 ``--x`` ``--y`` ``--z`` ``--yaw`` 覆盖。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_PKG_WORLDS_DIR = os.path.join(_REPO_ROOT, "ros2_ws", "src", "rl_car_gazebo", "worlds")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from curriculum_maps import (
    DEFAULT_ARENA_SPAWN_X,
    DEFAULT_ARENA_SPAWN_Y,
    DEFAULT_ARENA_SPAWN_YAW,
    DEFAULT_ARENA_SPAWN_Z,
    WORLD_BY_LEVEL,
    clamp_level,
)


def _available_worlds() -> list[str]:
    if not os.path.isdir(_PKG_WORLDS_DIR):
        return []
    out: list[str] = []
    for fn in sorted(os.listdir(_PKG_WORLDS_DIR)):
        if fn.endswith(".world"):
            out.append(fn)
    return out


def _normalize_world_arg(map_name: str) -> str:
    s = (map_name or "").strip()
    if not s:
        raise SystemExit("map 不能为空；或使用 --level 0~5，例如：python3 show_map.py --level 1")
    if s.endswith(".world"):
        return s
    return f"{s}.world"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="show_map.py",
        description="启动 Gazebo 显示 world；可用地图名或 --level（与课程一致）。",
    )
    ap.add_argument(
        "map",
        type=str,
        nargs="?",
        default="",
        help="地图文件名（不含路径），例如 level1_arena_8x8；与 --level 二选一",
    )
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="课程阶段 0~5：自动选择对应 arena world 与默认出生位姿（8×8 场地）",
    )
    ap.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--server", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--use-sim-time", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--x", type=float, default=None, help="spawn x（默认：无 --level 时为 0；有 --level 时为 1）")
    ap.add_argument("--y", type=float, default=None, help="spawn y（同上）")
    ap.add_argument("--z", type=float, default=None, help="spawn z（默认 0.1）")
    ap.add_argument("--yaw", type=float, default=None, help="spawn yaw rad（有 --level 时默认朝向对角目标）")

    args = ap.parse_args(argv)

    if shutil.which("ros2") is None:
        raise SystemExit(
            "未找到 ros2 命令。请先安装 ROS2，并在当前终端 source 对应 setup.bash。"
        )

    use_level = args.level is not None
    map_arg = (args.map or "").strip()

    if use_level and map_arg:
        raise SystemExit("请只指定其一：地图名（位置参数 map）或 --level，不要同时使用。")

    if use_level:
        lev = clamp_level(int(args.level))
        world = WORLD_BY_LEVEL[lev]
        x = float(DEFAULT_ARENA_SPAWN_X if args.x is None else args.x)
        y = float(DEFAULT_ARENA_SPAWN_Y if args.y is None else args.y)
        z = float(DEFAULT_ARENA_SPAWN_Z if args.z is None else args.z)
        yaw = float(DEFAULT_ARENA_SPAWN_YAW if args.yaw is None else args.yaw)
    else:
        if not map_arg:
            raise SystemExit(
                "请提供地图名，例如：python3 show_map.py level1_arena_8x8\n"
                "或使用课程阶段：python3 show_map.py --level 1"
            )
        world = _normalize_world_arg(map_arg)
        x = float(0.0 if args.x is None else args.x)
        y = float(0.0 if args.y is None else args.y)
        z = float(0.1 if args.z is None else args.z)
        yaw = float(0.0 if args.yaw is None else args.yaw)

    avail = _available_worlds()
    if world not in avail:
        hint = "\n".join([f"- {x}" for x in avail]) or "(worlds 目录为空)"
        raise SystemExit(
            f"未找到 world: {world}\n"
            f"期望位于：{_PKG_WORLDS_DIR}\n"
            f"当前可用 world：\n{hint}\n"
            "提示：请先 colcon build rl_car_gazebo；world 文件在 ros2_ws/src/rl_car_gazebo/worlds/"
        )

    cmd = [
        "ros2",
        "launch",
        "rl_car_gazebo",
        "sim.launch.py",
        f"world:={world}",
        f"gui:={args.gui}",
        f"server:={args.server}",
        f"use_sim_time:={args.use_sim_time}",
        f"x:={x}",
        f"y:={y}",
        f"z:={z}",
        f"yaw:={yaw}",
    ]

    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
