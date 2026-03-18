from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

try:
    from .camera_node import CameraNode, CameraState, ensure_rclpy_init as _ensure_cam_init
    from .lidar_node import LidarNode, LidarState, ensure_rclpy_init as _ensure_lidar_init
except ImportError:  # 允许直接 `python sensor_bridge.py`
    from camera_node import CameraNode, CameraState, ensure_rclpy_init as _ensure_cam_init  # type: ignore
    from lidar_node import LidarNode, LidarState, ensure_rclpy_init as _ensure_lidar_init  # type: ignore


@dataclass
class RobotState:
    stamp_sec: float = 0.0
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    position_xy: Optional[np.ndarray] = None  # (2,)
    yaw: Optional[float] = None
    linear_xy: Optional[np.ndarray] = None  # (2,)
    angular_z: Optional[float] = None
    lidar: Optional[LidarState] = None
    camera: Optional[CameraState] = None


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw from quaternion (Z axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class _OdomNode(Node):
    def __init__(self, topic: str = "/odom", node_name: str = "rl_car_odom_bridge"):
        super().__init__(node_name)
        self._lock = threading.Lock()
        self._latest: Optional[Odometry] = None
        self._sub = self.create_subscription(Odometry, topic, self._cb, 10)

    def _cb(self, msg: Odometry) -> None:
        with self._lock:
            self._latest = msg

    def get_latest(self) -> Optional[Odometry]:
        with self._lock:
            return self._latest


class SensorBridge:
    """
    统一传感器接口：state = SensorBridge(...).get_state()

    Topics 来源于你的 xacro:
    - lidar: libgazebo_ros_ray_sensor.so  remap ~/out:=scan  -> /scan
    - depth camera: namespace depth_cam + remap
        ~/image:=color_image      -> /depth_cam/color_image
        ~/depth/image:=depth_image -> /depth_cam/depth_image
        ~/points:=pointcloud      -> /depth_cam/pointcloud
    - odom: gazebo_ros_diff_drive 默认发布 /odom
    """

    def __init__(
        self,
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        color_topic: str = "/depth_cam/camera/image_raw",
        depth_topic: str = "/depth_cam/camera/depth/image_raw",
        points_topic: str = "/depth_cam/camera/points",
        subscribe_points: bool = False,
        spin_in_background: bool = True,
    ):
        _ensure_lidar_init()
        _ensure_cam_init()
        ensure_rclpy_init()

        self._lidar = LidarNode(topic=scan_topic)
        self._camera = CameraNode(
            color_topic=color_topic,
            depth_topic=depth_topic,
            points_topic=points_topic,
            subscribe_points=subscribe_points,
        )
        self._odom = _OdomNode(topic=odom_topic)

        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self._lidar)
        self._executor.add_node(self._camera)
        self._executor.add_node(self._odom)

        self._spin_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        if spin_in_background:
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()

    def _spin(self) -> None:
        while rclpy.ok() and not self._stop_evt.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def close(self) -> None:
        self._stop_evt.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        for n in (self._lidar, self._camera, self._odom):
            try:
                self._executor.remove_node(n)
            except Exception:
                pass
            try:
                n.destroy_node()
            except Exception:
                pass

    def get_state(self) -> RobotState:
        odom = self._odom.get_latest()
        lidar_state = self._lidar.get_state()
        cam_state = self._camera.get_state()

        s = RobotState(lidar=lidar_state, camera=cam_state)

        if odom is None:
            return s

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        t = odom.twist.twist

        s.stamp_sec = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        s.odom_frame = odom.header.frame_id or "odom"
        s.base_frame = odom.child_frame_id or "base_link"
        s.position_xy = np.asarray([p.x, p.y], dtype=np.float32)
        s.yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        s.linear_xy = np.asarray([t.linear.x, t.linear.y], dtype=np.float32)
        s.angular_z = float(t.angular.z)
        return s


def ensure_rclpy_init() -> None:
    if not rclpy.ok():
        rclpy.init(args=None)

if __name__ == "__main__":
    print("Starting Sensor Bridge")
    rclpy.init()
    sensor_bridge = SensorBridge()
    sensor_bridge.get_state()