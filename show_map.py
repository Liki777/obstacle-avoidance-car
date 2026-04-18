#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Iterable


_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_PKG_WORLDS_DIR = os.path.join(_REPO_ROOT, "ros2_ws", "src", "rl_car_gazebo", "worlds")


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
        raise SystemExit("map 不能为空，例如：python3 show_map.py level1_map")
    if s.endswith(".world"):
        return s
    return f"{s}.world"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="show_map.py",
        description="通用地图查看入口：show_map.py [map] -> 启动 Gazebo 显示对应 world",
    )
    ap.add_argument("map", type=str, help="地图名（例如 level1_map 或 level1_map.world）")
    ap.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--server", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--use-sim-time", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=0.1)
    ap.add_argument("--yaw", type=float, default=0.0)
    args = ap.parse_args(argv)

    if shutil.which("ros2") is None:
        raise SystemExit(
            "未找到 ros2 命令。请先安装 ROS2，并在当前终端 source 对应 setup.bash。"
        )

    world = _normalize_world_arg(args.map)
    avail = _available_worlds()
    if world not in avail:
        hint = "\n".join([f"- {x}" for x in avail]) or "(worlds 目录为空)"
        raise SystemExit(
            f"未找到 world: {world}\n"
            f"期望位于：{_PKG_WORLDS_DIR}\n"
            f"当前可用 world：\n{hint}\n"
            "提示：world 文件应放在 ros2_ws/src/rl_car_gazebo/worlds/（不要堆到 maps/）。"
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
        f"x:={args.x}",
        f"y:={args.y}",
        f"z:={args.z}",
        f"yaw:={args.yaw}",
    ]

    # 直接把控制权交给 ros2 launch（便于 Ctrl+C 停止）
    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

