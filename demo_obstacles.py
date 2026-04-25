#!/usr/bin/env python3
"""
课程 Level0~5 障碍展示（不训练）：按 level 选用对应 Gazebo world，行为与 ``map_design_document.md`` / ``train_ppo --level`` 对齐。

- **Level0**：无障碍；仅周期将车放回起点（可选）。
- **Level1**：固定 3×3 共 9 个混合障碍（扁圆柱/方柱/竖圆柱，与训练一致），**不随周期刷新**。
- **Level2**：随机静态障碍，按 ``--period`` **定时清空并重采样**。
- **Level3**：固定轨迹动态障碍（正弦/圆周），障体持续运动；每个 ``--period`` 内积分展示一段，随后回到起点并重置相位（同一套固定参数）。
- **Level4**：随机动态障碍，按 ``--period`` **定时重采样轨迹并重置**。
- **Level5**：随机静态 + 随机动态，按 ``--period`` **定时全套重采样**。

用法（需已 colcon build 并 ``source ros2_ws/install/setup.bash``；推荐与 ``run.py`` 统一）::

    python3 run.py demo --level 1
    python3 demo_obstacles.py --level 2 --period 8 --no-launch

场地默认 8×8（约 x,y∈[0,8]），起点 (1,1)、终点 (7,7)。
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

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from curriculum_maps import WORLD_BY_LEVEL, clamp_level

START_XY = (1.0, 1.0)
GOAL_XY = (7.0, 7.0)
SPAWN_Z = 0.1
MODEL_NAME = "rl_car"

# 与 map_design_document / train_ppo 默认一致
MAP_X_MIN = -0.25
MAP_X_MAX = 8.25
MAP_Y_MIN = -0.25
MAP_Y_MAX = 8.25


def _yaw_towards_goal() -> float:
    dx = GOAL_XY[0] - START_XY[0]
    dy = GOAL_XY[1] - START_XY[1]
    return float(math.atan2(dy, dx))


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


def _run_dynamic_tick_loop(
    *,
    mgr: Any,
    node: Any,
    cmd_pub: Any,
    specs: list[Any],
    duration_sec: float,
    dyn_dt: float,
    pose_fn: Any,
) -> None:
    """在给定时间内按 dyn_dt 更新动态障碍位姿。"""
    import rclpy

    t_sim = 0.0
    t_end = time.time() + float(duration_sec)
    while time.time() < t_end and rclpy.ok():
        for sp in specs:
            x, y, yaw = pose_fn(sp, t_sim)
            mgr.set_entity_pose(name=sp.name, x=x, y=y, yaw=yaw)
        _stop_robot(node, cmd_pub)
        _pump(node, 8, 0.015)
        t_sim += float(dyn_dt)
        time.sleep(float(dyn_dt))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="课程 Level0~5 障碍展示（不训练）。Level1 固定障碍不刷新；Level2/4/5 按 period 刷新。"
    )
    ap.add_argument("--level", type=int, default=2, help="课程阶段 0~5（与 train_ppo --level 含义一致）")
    ap.add_argument(
        "--period",
        type=float,
        default=8.0,
        help="Level0 仅重置小车；Level2/4/5：障碍刷新周期(s)；Level3：单段动态演示时长后重复",
    )
    ap.add_argument("--dyn-dt", type=float, default=0.05, help="Level3~5 动态障碍位姿更新步长(s)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--gui", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--use-sim-time", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args(argv)

    if which("ros2") is None:
        print("未找到 ros2 命令：请先安装 ROS2 并 source ros2_ws/install/setup.bash", file=sys.stderr)
        return 1

    lev = clamp_level(args.level)
    if int(args.level) != lev:
        print(f"[INFO] level 已限制到 0~5，当前使用 {lev}", flush=True)

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    rng_np = np.random.default_rng(args.seed)

    world = WORLD_BY_LEVEL[lev]
    sx, sy = START_XY
    yaw0 = _yaw_towards_goal()

    launch_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_launch:
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
        time.sleep(12.0)

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Twist

    rclpy.init(args=None)
    node = Node("demo_obstacles")
    goal_pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
    cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)

    from obstacle_environment.scenario_manager import GazeboObstacleManager, GazeboObstacleManagerConfig
    from obstacle_environment.world_generator.dynamic_obstacle_presets import (
        builtin_fixed_dynamic_specs,
        builtin_level1_fixed_mixed_3x3,
        pose_at_time,
        sample_random_dynamic_specs,
    )
    from obstacle_environment.world_generator.static_obstacles import (
        StaticObstacleSamplingConfig,
        sample_static_obstacles,
    )

    mgr = GazeboObstacleManager(node=node, cfg=GazeboObstacleManagerConfig())

    def _shutdown(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cycle = 0

    def _robot_and_goal() -> None:
        _stop_robot(node, cmd_pub)
        _try_set_robot_pose(node, x=sx, y=sy, z=SPAWN_Z, yaw=yaw0)
        _publish_goal(node, goal_pub, GOAL_XY[0], GOAL_XY[1])

    try:
        # ========== Level0：无障碍，仅周期回起点 ==========
        if lev == 0:
            node.get_logger().info("Level0：无障碍。按 Ctrl+C 结束。")
            if not _try_set_robot_pose(node, x=sx, y=sy, z=SPAWN_Z, yaw=yaw0):
                node.get_logger().warn("SetEntityState 不可用：小车可能未回到起点。")
            _publish_goal(node, goal_pub, GOAL_XY[0], GOAL_XY[1])
            while rclpy.ok():
                t0 = time.time()
                while time.time() - t0 < float(args.period) and rclpy.ok():
                    _pump(node, 6, 0.05)
                    _stop_robot(node, cmd_pub)
                _stop_robot(node, cmd_pub)
                _try_set_robot_pose(node, x=sx, y=sy, z=SPAWN_Z, yaw=yaw0)
                _publish_goal(node, goal_pub, GOAL_XY[0], GOAL_XY[1])

        # ========== Level1：固定静态，障碍只刷一次 ==========
        elif lev == 1:
            mgr.clear_spawned()
            _pump(node, 20, 0.02)
            _robot_and_goal()
            specs_l1 = builtin_level1_fixed_mixed_3x3(name_prefix="demo_l1")
            mgr.spawn_mixed_static(specs=specs_l1)
            node.get_logger().info(
                f"Level1：已生成 {len(specs_l1)} 个固定混合障碍（3×3，不再刷新）。按 Ctrl+C 结束。"
            )
            while rclpy.ok():
                _pump(node, 12, 0.05)
                _stop_robot(node, cmd_pub)

        # ========== Level2：随机静态，定时全量重刷 ==========
        elif lev == 2:
            while rclpy.ok():
                mgr.clear_spawned()
                _pump(node, 15, 0.02)
                _robot_and_goal()
                samp = StaticObstacleSamplingConfig(
                    x_min=float(MAP_X_MIN),
                    x_max=float(MAP_X_MAX),
                    y_min=float(MAP_Y_MIN),
                    y_max=float(MAP_Y_MAX),
                    min_dist_to_robot=1.5,
                    min_dist_to_goal=1.5,
                    min_dist_between_obstacles=1.0,
                )
                n_obs = int(rng_np.integers(3, 6))
                specs_s = sample_static_obstacles(
                    num_obstacles=n_obs,
                    start_xy=(sx, sy),
                    goal_xy=GOAL_XY,
                    cfg=samp,
                    rng=rng_np,
                    name_prefix=f"demo_r2_c{cycle}_",
                )
                obs_boxes = [(s.name, float(s.x), float(s.y), float(s.yaw)) for s in specs_s]
                mgr.spawn_static_boxes(obstacles=obs_boxes)
                node.get_logger().info(
                    f"Level2 周期 {cycle}: {len(obs_boxes)} 个随机静态障碍；{args.period:g}s 后刷新。"
                )
                cycle += 1
                t0 = time.time()
                while time.time() - t0 < float(args.period) and rclpy.ok():
                    _pump(node, 6, 0.05)
                    _stop_robot(node, cmd_pub)

        # ========== Level3：固定轨迹动态，障碍每 period 段内连续运动 ==========
        elif lev == 3:
            dyn_specs = list(builtin_fixed_dynamic_specs(name_prefix="demo_dyn"))
            while rclpy.ok():
                mgr.clear_spawned()
                _pump(node, 15, 0.02)
                _robot_and_goal()
                spawn_list = [(s.name, *pose_at_time(s, 0.0)) for s in dyn_specs]
                mgr.spawn_static_boxes(obstacles=spawn_list)
                node.get_logger().info(
                    f"Level3：固定动态障碍 {len(dyn_specs)} 个；演示 {args.period:g}s 后重复同轨迹。"
                )
                _run_dynamic_tick_loop(
                    mgr=mgr,
                    node=node,
                    cmd_pub=cmd_pub,
                    specs=dyn_specs,
                    duration_sec=float(args.period),
                    dyn_dt=float(args.dyn_dt),
                    pose_fn=pose_at_time,
                )

        # ========== Level4：随机动态，定时重采样 ==========
        elif lev == 4:
            while rclpy.ok():
                mgr.clear_spawned()
                _pump(node, 15, 0.02)
                _robot_and_goal()
                n_dyn = int(rng.randint(1, 3))
                rng_d = random.Random((args.seed or 0) ^ (cycle * 7919))
                dyn_specs = sample_random_dynamic_specs(
                    rng_d,
                    count=n_dyn,
                    map_x_min=MAP_X_MIN,
                    map_x_max=MAP_X_MAX,
                    map_y_min=MAP_Y_MIN,
                    map_y_max=MAP_Y_MAX,
                    start_xy=(sx, sy),
                    goal_xy=GOAL_XY,
                    name_prefix=f"demo_rd4_{cycle}_",
                )
                if dyn_specs:
                    spawn_list = [(s.name, *pose_at_time(s, 0.0)) for s in dyn_specs]
                    mgr.spawn_static_boxes(obstacles=spawn_list)
                node.get_logger().info(
                    f"Level4 周期 {cycle}: {len(dyn_specs)} 个随机动态障碍；演示 {args.period:g}s。"
                )
                cycle += 1
                _run_dynamic_tick_loop(
                    mgr=mgr,
                    node=node,
                    cmd_pub=cmd_pub,
                    specs=dyn_specs,
                    duration_sec=float(args.period),
                    dyn_dt=float(args.dyn_dt),
                    pose_fn=pose_at_time,
                )

        # ========== Level5：随机静 + 动 ==========
        else:
            while rclpy.ok():
                mgr.clear_spawned()
                _pump(node, 15, 0.02)
                _robot_and_goal()
                samp = StaticObstacleSamplingConfig(
                    x_min=float(MAP_X_MIN),
                    x_max=float(MAP_X_MAX),
                    y_min=float(MAP_Y_MIN),
                    y_max=float(MAP_Y_MAX),
                    min_dist_to_robot=1.5,
                    min_dist_to_goal=1.5,
                    min_dist_between_obstacles=1.0,
                )
                n_s = int(rng_np.integers(3, 6))
                specs_s = sample_static_obstacles(
                    num_obstacles=n_s,
                    start_xy=(sx, sy),
                    goal_xy=GOAL_XY,
                    cfg=samp,
                    rng=rng_np,
                    name_prefix=f"demo_r5s_{cycle}_",
                )
                obs_boxes = [(s.name, float(s.x), float(s.y), float(s.yaw)) for s in specs_s]
                if obs_boxes:
                    mgr.spawn_static_boxes(obstacles=obs_boxes)

                n_dyn = int(rng.randint(1, 2))
                rng_d = random.Random((args.seed or 0) ^ (cycle * 11003))
                dyn_specs = sample_random_dynamic_specs(
                    rng_d,
                    count=n_dyn,
                    map_x_min=MAP_X_MIN,
                    map_x_max=MAP_X_MAX,
                    map_y_min=MAP_Y_MIN,
                    map_y_max=MAP_Y_MAX,
                    start_xy=(sx, sy),
                    goal_xy=GOAL_XY,
                    name_prefix=f"demo_rd5_{cycle}_",
                )
                if dyn_specs:
                    spawn_list = [(s.name, *pose_at_time(s, 0.0)) for s in dyn_specs]
                    mgr.spawn_static_boxes(obstacles=spawn_list)

                node.get_logger().info(
                    f"Level5 周期 {cycle}: 静态 {len(obs_boxes)} + 动态 {len(dyn_specs)}；演示 {args.period:g}s。"
                )
                cycle += 1
                _run_dynamic_tick_loop(
                    mgr=mgr,
                    node=node,
                    cmd_pub=cmd_pub,
                    specs=dyn_specs,
                    duration_sec=float(args.period),
                    dyn_dt=float(args.dyn_dt),
                    pose_fn=pose_at_time,
                )

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
