from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(math.sin(float(yaw) * 0.5))
    q.w = float(math.cos(float(yaw) * 0.5))
    return q


def _simple_box_sdf(*, size_xy: float, size_z: float) -> str:
    """
    生成一个最小可用的 box SDF（用于 Level1 静态障碍）。
    不依赖 maps/ 资源文件，避免把逻辑/资产堆在 maps 目录。
    """
    sx = float(size_xy)
    sy = float(size_xy)
    sz = float(size_z)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="box_obstacle">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>
        </geometry>
        <material>
          <ambient>0.3 0.3 0.3 1</ambient>
          <diffuse>0.6 0.6 0.6 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def _box_sdf_general(*, sx: float, sy: float, sz: float) -> str:
    """任意长方体 SDF（用于立方体、薄墙等）。"""
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="mixed_box_obstacle">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>{float(sx):.4f} {float(sy):.4f} {float(sz):.4f}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{float(sx):.4f} {float(sy):.4f} {float(sz):.4f}</size></box>
        </geometry>
        <material>
          <ambient>0.32 0.32 0.34 1</ambient>
          <diffuse>0.55 0.55 0.58 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def _cylinder_sdf(*, radius: float, length: float) -> str:
    """扁圆柱：圆柱轴向为 Z，length 很小时为扁盘。"""
    r = float(radius)
    h = float(length)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="mixed_cylinder_obstacle">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder>
        </geometry>
        <material>
          <ambient>0.28 0.42 0.55 1</ambient>
          <diffuse>0.35 0.55 0.72 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


@dataclass
class MixedObstacleSpec:
    """
    混合静态障碍（由 ``GazeboObstacleManager.spawn_mixed_static`` 生成）。

    - ``cube``：``sx, sy, sz`` 为 full box size（米）。
    - ``cylinder``：``sx``=半径，``sy``=轴向高度（扁盘取小值），``sz`` 忽略。
    - ``wall``：``sx``=面内长边，``sy``=厚度，``sz``=高度；``yaw`` 旋转后面向可随机。
    """

    name: str
    kind: Literal["cube", "cylinder", "wall"]
    x: float
    y: float
    yaw: float
    sx: float = 0.42
    sy: float = 0.42
    sz: float = 0.5


@dataclass
class GazeboObstacleManagerConfig:
    """
    Gazebo Classic（gazebo_ros_pkgs）障碍物 spawn/delete 管理配置。
    若服务不可用，该管理器会自动降级为空操作（不影响训练主流程）。
    """

    # 服务名候选：不同启动方式/命名空间下可能不同
    spawn_entity_services: tuple[str, ...] = ("/spawn_entity", "/gazebo/spawn_entity")
    delete_entity_services: tuple[str, ...] = ("/delete_entity", "/gazebo/delete_entity")

    wait_each_service_sec: float = 2.0

    # 生成的 box 尺寸（m）
    obstacle_size_xy: float = 0.45
    obstacle_size_z: float = 0.7
    obstacle_z: float = 0.0


class GazeboObstacleManager:
    """
    管理一组“由训练系统生成”的障碍物实体：
    - reset 时：先 delete 上一轮生成的，再 spawn 新的一组
    """

    def __init__(self, *, node: Any, cfg: GazeboObstacleManagerConfig | None = None) -> None:
        self._node = node
        self._cfg = cfg or GazeboObstacleManagerConfig()

        self._spawn_cli: Any = None
        self._delete_cli: Any = None

        self._spawn_disabled = False
        self._delete_disabled = False

        self._spawned_names: list[str] = []

        # lazy import：没有 gazebo_msgs 时直接降级
        try:
            from gazebo_msgs.srv import SpawnEntity, DeleteEntity  # type: ignore[import-untyped]  # noqa: F401
        except Exception as e:
            self._spawn_disabled = True
            self._delete_disabled = True
            try:
                self._node.get_logger().warn(f"gazebo_msgs 不可用，障碍物 spawn/delete 将跳过: {e}")
            except Exception:
                pass

    def _pump(self, n: int = 20, timeout_sec: float = 0.02) -> None:
        try:
            import rclpy

            for _ in range(int(n)):
                rclpy.spin_once(self._node, timeout_sec=float(timeout_sec))
        except Exception:
            pass

    def _resolve_clients(self) -> None:
        if self._spawn_disabled and self._delete_disabled:
            return
        try:
            from gazebo_msgs.srv import SpawnEntity, DeleteEntity  # type: ignore[import-untyped]
        except Exception:
            self._spawn_disabled = True
            self._delete_disabled = True
            return

        self._pump(15, 0.02)

        if (self._spawn_cli is None) and (not self._spawn_disabled):
            for svc in self._cfg.spawn_entity_services:
                cli = self._node.create_client(SpawnEntity, svc)
                t0 = time.time()
                while time.time() - t0 < float(self._cfg.wait_each_service_sec):
                    if cli.service_is_ready():
                        self._spawn_cli = cli
                        try:
                            self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（SpawnEntity）")
                        except Exception:
                            pass
                        break
                    self._pump(10, 0.02)
                if self._spawn_cli is not None:
                    break
                try:
                    self._node.destroy_client(cli)
                except Exception:
                    pass
            if self._spawn_cli is None:
                self._spawn_disabled = True
                try:
                    self._node.get_logger().warn(
                        "未连接到 SpawnEntity 服务（将跳过障碍物生成）。"
                    )
                except Exception:
                    pass

        if (self._delete_cli is None) and (not self._delete_disabled):
            for svc in self._cfg.delete_entity_services:
                cli = self._node.create_client(DeleteEntity, svc)
                t0 = time.time()
                while time.time() - t0 < float(self._cfg.wait_each_service_sec):
                    if cli.service_is_ready():
                        self._delete_cli = cli
                        try:
                            self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（DeleteEntity）")
                        except Exception:
                            pass
                        break
                    self._pump(10, 0.02)
                if self._delete_cli is not None:
                    break
                try:
                    self._node.destroy_client(cli)
                except Exception:
                    pass
            if self._delete_cli is None:
                self._delete_disabled = True
                try:
                    self._node.get_logger().warn(
                        "未连接到 DeleteEntity 服务（将无法清理上一轮障碍物）。"
                    )
                except Exception:
                    pass

    def clear_spawned(self) -> None:
        self._resolve_clients()
        if self._delete_cli is None:
            self._spawned_names = []
            return
        try:
            from gazebo_msgs.srv import DeleteEntity  # type: ignore[import-untyped]
            import rclpy

            for name in list(self._spawned_names):
                req = DeleteEntity.Request()
                req.name = str(name)
                fut = self._delete_cli.call_async(req)
                rclpy.spin_until_future_complete(self._node, fut, timeout_sec=2.0)
            self._spawned_names = []
        except Exception:
            self._spawned_names = []

    def spawn_static_boxes(self, *, obstacles: list[tuple[str, float, float, float]]) -> None:
        """
        obstacles: [(name, x, y, yaw), ...]
        """
        self._resolve_clients()
        if self._spawn_cli is None:
            return
        try:
            from gazebo_msgs.srv import SpawnEntity  # type: ignore[import-untyped]
            import rclpy

            sdf = _simple_box_sdf(
                size_xy=float(self._cfg.obstacle_size_xy),
                size_z=float(self._cfg.obstacle_size_z),
            )
            for (name, x, y, yaw) in obstacles:
                req = SpawnEntity.Request()
                req.name = str(name)
                req.xml = str(sdf)
                # 一些实现会忽略 initial_pose.frame_id，这里尽量保持简单
                p = Pose()
                p.position.x = float(x)
                p.position.y = float(y)
                p.position.z = float(self._cfg.obstacle_z)
                p.orientation = _yaw_to_quat(float(yaw))
                req.initial_pose = p
                fut = self._spawn_cli.call_async(req)
                rclpy.spin_until_future_complete(self._node, fut, timeout_sec=3.0)
                self._spawned_names.append(str(name))
        except Exception:
            return

    def spawn_mixed_static(self, *, specs: list[MixedObstacleSpec]) -> None:
        """
        生成立方体 / 扁圆柱 / 薄墙（墙为水平薄盒，绕 yaw 旋转后宽面可朝向任意水平方向）。
        """
        self._resolve_clients()
        if self._spawn_cli is None:
            return
        try:
            from gazebo_msgs.srv import SpawnEntity  # type: ignore[import-untyped]
            import rclpy

            for sp in specs:
                k = str(sp.kind)
                if k == "cube":
                    sdf = _box_sdf_general(sx=float(sp.sx), sy=float(sp.sy), sz=float(sp.sz))
                    zc = float(sp.sz) * 0.5 + 1e-3
                elif k == "cylinder":
                    sdf = _cylinder_sdf(radius=float(sp.sx), length=float(sp.sy))
                    zc = float(sp.sy) * 0.5 + 1e-3
                elif k == "wall":
                    sdf = _box_sdf_general(sx=float(sp.sx), sy=float(sp.sy), sz=float(sp.sz))
                    zc = float(sp.sz) * 0.5 + 1e-3
                else:
                    continue

                req = SpawnEntity.Request()
                req.name = str(sp.name)
                req.xml = str(sdf)
                p = Pose()
                p.position.x = float(sp.x)
                p.position.y = float(sp.y)
                p.position.z = float(zc)
                p.orientation = _yaw_to_quat(float(sp.yaw))
                req.initial_pose = p
                fut = self._spawn_cli.call_async(req)
                rclpy.spin_until_future_complete(self._node, fut, timeout_sec=3.0)
                self._spawned_names.append(str(sp.name))
        except Exception:
            return

