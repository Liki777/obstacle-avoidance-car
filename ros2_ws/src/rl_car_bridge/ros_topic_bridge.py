from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

try:
    from .conversions import image_to_numpy, laserscan_to_numpy
except ImportError:
    from conversions import image_to_numpy, laserscan_to_numpy  # type: ignore


def ensure_rclpy_init() -> None:
    if not rclpy.ok():
        rclpy.init(args=None)


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@dataclass
class BridgeState:
    stamp_sec: float = 0.0
    position_xy: Optional[np.ndarray] = None  # (2,) float32
    yaw: Optional[float] = None
    linear_xy: Optional[np.ndarray] = None  # (2,) float32
    angular_z: Optional[float] = None
    lidar_ranges: Optional[np.ndarray] = None  # (N,) float32
    color_image: Optional[np.ndarray] = None  # HxWx3 uint8
    depth_image: Optional[np.ndarray] = None  # HxW float32/uint16


class RosTopicBridge(Node):
    """
    把 ROS2 topics 转成 Python 数据（numpy），方便 Gym 环境直接调用。

    默认 topics 与你当前 xacro 一致：
    - /scan
    - /odom
    - /depth_cam/camera/image_raw
    - /depth_cam/camera/depth/image_raw
    """

    def __init__(
        self,
        *,
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        color_topic: str = "/depth_cam/camera/image_raw",
        depth_topic: str = "/depth_cam/camera/depth/image_raw",
        node_name: str = "rl_car_ros_topic_bridge",
        spin_in_background: bool = True,
        qos: int = 10,
    ):
        ensure_rclpy_init()
        super().__init__(node_name)

        self._lock = threading.Lock()
        self._state = BridgeState()

        self._sub_scan = self.create_subscription(LaserScan, scan_topic, self._cb_scan, qos)
        self._sub_odom = self.create_subscription(Odometry, odom_topic, self._cb_odom, qos)
        self._sub_color = self.create_subscription(Image, color_topic, self._cb_color, qos)
        self._sub_depth = self.create_subscription(Image, depth_topic, self._cb_depth, qos)

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)

        self._stop_evt = threading.Event()
        self._spin_thread: Optional[threading.Thread] = None
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
        try:
            self._executor.remove_node(self)
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass

    def _cb_scan(self, msg: LaserScan) -> None:
        ranges = laserscan_to_numpy(msg, clip=True)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._state.lidar_ranges = ranges
            self._state.stamp_sec = max(self._state.stamp_sec, stamp)

    def _cb_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._state.position_xy = np.asarray([p.x, p.y], dtype=np.float32)
            self._state.yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
            self._state.linear_xy = np.asarray([t.linear.x, t.linear.y], dtype=np.float32)
            self._state.angular_z = float(t.angular.z)
            self._state.stamp_sec = max(self._state.stamp_sec, stamp)

    def _cb_color(self, msg: Image) -> None:
        img = image_to_numpy(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._state.color_image = img
            self._state.stamp_sec = max(self._state.stamp_sec, stamp)

    def _cb_depth(self, msg: Image) -> None:
        img = image_to_numpy(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._state.depth_image = img
            self._state.stamp_sec = max(self._state.stamp_sec, stamp)

    def get_state(self) -> BridgeState:
        with self._lock:
            return self._state


if __name__ == "__main__":
    b = RosTopicBridge()
    try:
        while True:
            s = b.get_state()
            print(
                f"stamp={s.stamp_sec:.3f} "
                f"scan={'Y' if s.lidar_ranges is not None else 'N'} "
                f"odom={'Y' if s.position_xy is not None else 'N'} "
                f"rgb={'Y' if s.color_image is not None else 'N'} "
                f"depth={'Y' if s.depth_image is not None else 'N'}"
            )
            import time

            time.sleep(0.5)
    finally:
        b.close()

