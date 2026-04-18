#!/usr/bin/env python3
"""
8×8 场地随机障碍展示（不训练）：定时清理旧障碍 → 随机生成立方体/扁圆柱/薄墙 → 重置小车与目标。

用法（需已 colcon build 并 source ros2_ws/install/setup.bash）::

    # 一键启动仿真 + 展示循环（阶段 1/2/3 对应不同 world 与障碍密度）
    python3 demo_random_obstacles.py --level 1

    # 仅连接已运行的 Gazebo（另开终端先 ros2 launch ...）
    python3 demo_random_obstacles.py --level 2 --no-launch

阶段 world：``level1_arena_8x8.world`` / ``level2_arena_8x8.world`` / ``level3_arena_8x8.world``。
场地内沿约 x,y∈[0,8]，起点 (1,1)、终点 (7,7)。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import subprocess
import sys
import time
from shutil import which
from typing import Any, Optional

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- 8×8 场地与起终点（与 worlds/*_arena_8x8.world 一致）---
ARENA_XY = (0.0, 8.0)
START_XY = (1.0, 1.0)
GOAL_XY = (7.0, 7.0)
SPAWN_Z = 0.1
MODEL_NAME = "rl_car"

_LEVEL_WORLDS = {
    1: "level1_arena_8x8.world",
    2: "level2_arena_8x8.world",
    3: "level3_arena_8x8.world",
}


def _yaw_towards_goal() -> float:
    dx = GOAL_XY[0] - START_XY[0]
    dy = GOAL_XY[1] - START_XY[1]
    return float(math.atan2(dy, dx))


def _footprint_radius(kind: str, sx: float, sy: float) -> float:
    if kind == "cylinder":
        return float(sx) * 1.08
    return 0.5 * math.hypot(float(sx), float(sy)) + 1e-3


def _level_obstacle_count(rng: random.Random, level: int) -> int:
    if level <= 1:
        return rng.randint(4, 7)
    if level == 2:
        return rng.randint(7, 10)
    return rng.randint(11, 15)


def _pick_kind(rng: random.Random, level: int) -> str:
    if level <= 1:
        return rng.choices(["cube", "cylinder", "wall"], weights=[0.45, 0.35, 0.2], k=1)[0]
    if level == 2:
        return rng.choices(["cube", "cylinder", "wall"], weights=[0.32, 0.28, 0.4], k=1)[0]
    return rng.choices(["cube", "cylinder", "wall"], weights=[0.22, 0.22, 0.56], k=1)[0]


def _sample_dims(rng: random.Random, kind: str) -> tuple[float, float, float]:
    if kind == "cube":
        s = rng.uniform(0.34, 0.5)
        h = rng.uniform(0.4, 0.58)
        return s, s, h
    if kind == "cylinder":
        r = rng.uniform(0.22, 0.36)
        h = rng.uniform(0.06, 0.12)
        return r, h, 0.0
    length = rng.uniform(1.15, 2.05)
    thick = rng.uniform(0.08, 0.14)
    height = rng.uniform(0.42, 0.62)
    return length, thick, height


def _sample_obstacles(
    rng: random.Random,
    *,
    level: int,
    prefix: str,
) -> list[Any]:
    from obstacle_environment.scenario_manager.gazebo_obstacle_manager import MixedObstacleSpec

    n = _level_obstacle_count(rng, level)
    placed: list[tuple[float, float, float, str, float, float, float]] = []
    # (x, y, r_clear, kind, sx, sy, sz)
    specs: list[MixedObstacleSpec] = []
    margin = 0.42
    xmin, xmax = ARENA_XY[0] + margin, ARENA_XY[1] - margin
    ymin, ymax = ARENA_XY[0] + margin, ARENA_XY[1] - margin

    def ok_pose(x: float, y: float, r: float) -> bool:
        if math.hypot(x - START_XY[0], y - START_XY[1]) < r + 1.05:
            return False
        if math.hypot(x - GOAL_XY[0], y - GOAL_XY[1]) < r + 1.05:
            return False
        for px, py, pr, *_ in placed:
            if math.hypot(x - px, y - py) < r + pr + 0.12:
                return False
        return True

    for i in range(n):
        kind = _pick_kind(rng, level)
        sx, sy, sz = _sample_dims(rng, kind)
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        rfp = _footprint_radius(kind, sx, sy if kind != "cylinder" else sx)
        placed_ok = False
        for _ in range(90):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if ok_pose(x, y, rfp):
                placed.append((x, y, rfp, kind, sx, sy, sz))
                placed_ok = True
                break
        if not placed_ok:
            continue
        name = f"{prefix}_obs_{i:02d}"
        if kind == "cube":
            specs.append(MixedObstacleSpec(name=name, kind="cube", x=x, y=y, yaw=yaw, sx=sx, sy=sy, sz=sz))
        elif kind == "cylinder":
            specs.append(
                MixedObstacleSpec(name=name, kind="cylinder", x=x, y=y, yaw=yaw, sx=sx, sy=sy, sz=0.0)
            )
        else:
            specs.append(MixedObstacleSpec(name=name, kind="wall", x=x, y=y, yaw=yaw, sx=sx, sy=sy, sz=sz))
    return specs


def _pump(node: Any, n: int = 12, dt: float = 0.02) -> None:
    import rclpy

    for _ in range(int(n)):
        rclpy.spin_once(node, timeout_sec=float(dt))


def _try_set_robot_pose(node: Any, *, x: float, y: float, z: float, yaw: float) -> bool:
    from geometry_msgs.msg import Twist
    from gazebo_msgs.msg import EntityState, ModelState
    from gazebo_msgs.srv import SetEntityState, SetModelState

    try:
        from geometry_msgs.msg import Quaternion

        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = float(math.sin(float(yaw) * 0.5))
        q.w = float(math.cos(float(yaw) * 0.5))
    except Exception:
        return False

    for svc_name, use_entity in (
        ("/set_entity_state", True),
        ("/gazebo/set_entity_state", True),
        ("/set_model_state", False),
        ("/gazebo/set_model_state", False),
    ):
        if use_entity:
            cli = node.create_client(SetEntityState, svc_name)
        else:
            cli = node.create_client(SetModelState, svc_name)
        t0 = time.time()
        while time.time() - t0 < 1.2:
            if cli.service_is_ready():
                break
            _pump(node, 8, 0.02)
        if not cli.service_is_ready():
            try:
                node.destroy_client(cli)
            except Exception:
                pass
            continue
        try:
            import rclpy

            if use_entity:
                st = EntityState()
                st.name = MODEL_NAME
                st.pose.position.x = float(x)
                st.pose.position.y = float(y)
                st.pose.position.z = float(z)
                st.pose.orientation = q
                st.twist = Twist()
                st.reference_frame = "world"
                req = SetEntityState.Request()
                req.state = st
            else:
                ms = ModelState()
                ms.model_name = MODEL_NAME
                ms.pose.position.x = float(x)
                ms.pose.position.y = float(y)
                ms.pose.position.z = float(z)
                ms.pose.orientation = q
                ms.twist = Twist()
                ms.reference_frame = "world"
                req = SetModelState.Request()
                req.model_state = ms
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=4.0)
            res = fut.result()
            ok = res is not None and getattr(res, "success", True)
            try:
                node.destroy_client(cli)
            except Exception:
                pass
            if ok:
                return True
        except Exception:
            try:
                node.destroy_client(cli)
            except Exception:
                pass
    return False


def _publish_goal(node: Any, pub: Any, gx: float, gy: float) -> None:
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "odom"
    msg.pose.position.x = float(gx)
    msg.pose.position.y = float(gy)
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = 1.0
    pub.publish(msg)


def _stop_robot(node: Any, pub: Any) -> None:
    from geometry_msgs.msg import Twist

    pub.publish(Twist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="8×8 随机障碍展示（不训练），支持阶段 1/2/3。")
    ap.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="课程阶段（决定 world 与障碍数量）")
    ap.add_argument("--period", type=float, default=8.0, help="每次随机刷新间隔（秒）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（默认可复现性不保证）")
    ap.add_argument("--no-launch", action="store_true", help="不启动 launch，仅连接已有 Gazebo")
    ap.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--use-sim-time", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args(argv)

    if which("ros2") is None:
        print("未找到 ros2 命令：请先安装 ROS2 并 source ros2_ws/install/setup.bash", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    world = _LEVEL_WORLDS[int(args.level)]
    sx, sy = START_XY
    yaw0 = _yaw_towards_goal()

    launch_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_launch:
        if which("ros2") is None:
            print("未找到 ros2 命令", file=sys.stderr)
            return 1
        cmd = [
            "ros2",
            "launch",
            "rl_car_gazebo",
            "sim.launch.py",
            f"world:={world}",
            f"gui:={args.gui}",
            f"use_sim_time:={args.use_sim_time}",
            f"x:={sx}",
            f"y:={sy}",
            f"z:={SPAWN_Z}",
            f"yaw:={yaw0}",
        ]
        print("启动:", " ".join(cmd), flush=True)
        launch_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # sim.launch 中 spawn 延迟 6s，再加 gz 就绪裕量
        time.sleep(12.0)

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from geometry_msgs.msg import PoseStamped

    rclpy.init(args=None)
    node = Node("demo_random_obstacles")
    goal_pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
    cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)

    from obstacle_environment.scenario_manager import GazeboObstacleManager, GazeboObstacleManagerConfig

    mgr = GazeboObstacleManager(node=node, cfg=GazeboObstacleManagerConfig())
    cycle = 0

    def _shutdown(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while rclpy.ok():
            prefix = f"demo8_l{int(args.level)}_c{cycle}"
            specs = _sample_obstacles(rng, level=int(args.level), prefix=prefix)
            mgr.clear_spawned()
            _pump(node, 20, 0.02)
            _stop_robot(node, cmd_pub)
            ok = _try_set_robot_pose(node, x=sx, y=sy, z=SPAWN_Z, yaw=yaw0)
            if not ok and cycle == 0:
                node.get_logger().warn("SetModelState/SetEntityState 不可用：小车可能未回到起点。")
            _publish_goal(node, goal_pub, GOAL_XY[0], GOAL_XY[1])
            mgr.spawn_mixed_static(specs=specs)
            node.get_logger().info(
                f"周期 {cycle}: 已生成 {len(specs)} 个障碍（level={args.level}），{args.period:g}s 后刷新。"
            )
            cycle += 1
            t0 = time.time()
            while time.time() - t0 < float(args.period):
                _pump(node, 6, 0.05)
                _stop_robot(node, cmd_pub)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.clear_spawned()
        _stop_robot(node, cmd_pub)
        node.destroy_node()
        rclpy.shutdown()
        if launch_proc is not None:
            launch_proc.terminate()
            try:
                launch_proc.wait(timeout=5.0)
            except Exception:
                launch_proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
