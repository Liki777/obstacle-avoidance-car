from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

try:
    from sensor_msgs_py import point_cloud2 as pc2  # type: ignore
except Exception:  # pragma: no cover
    pc2 = None


@dataclass
class CameraState:
    stamp_sec: float = 0.0
    frame_id: str = ""
    color: Optional[np.ndarray] = None  # HxWx3 uint8
    depth: Optional[np.ndarray] = None  # HxW float32/uint16 depending on encoding
    pointcloud_xyz: Optional[np.ndarray] = None  # Nx3 float32


def _image_to_numpy(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = msg.data

    # Common encodings from gazebo_ros_camera: rgb8/bgr8/rgba8/bgraf8, 16UC1/32FC1 for depth
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = arr[..., ::-1]
        return arr
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))[..., :3]
        if enc == "bgra8":
            arr = arr[..., ::-1]
        return arr
    if enc in ("mono8",):
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    if enc in ("16uc1",):
        return np.frombuffer(data, dtype=np.uint16).reshape((h, w))
    if enc in ("32fc1",):
        return np.frombuffer(data, dtype=np.float32).reshape((h, w))

    # Fallback: treat as uint8 flat
    return np.frombuffer(data, dtype=np.uint8)


def _pc2_to_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
    if pc2 is None:
        return None
    pts = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        pts.append((float(p[0]), float(p[1]), float(p[2])))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


class CameraNode(Node):
    def __init__(
        self,
        color_topic: str = "/depth_cam/camera/image_raw",
        depth_topic: str = "/depth_cam/camera/depth/image_raw",
        points_topic: str = "/depth_cam/camera/points",
        node_name: str = "rl_car_camera_bridge",
        subscribe_points: bool = False,
    ):
        super().__init__(node_name)
        self._lock = threading.Lock()
        self._state = CameraState()

        self._sub_color = self.create_subscription(Image, color_topic, self._cb_color, 10)
        self._sub_depth = self.create_subscription(Image, depth_topic, self._cb_depth, 10)
        self._sub_points = None
        if subscribe_points:
            self._sub_points = self.create_subscription(PointCloud2, points_topic, self._cb_points, 3)

    def _cb_color(self, msg: Image) -> None:
        img = _image_to_numpy(msg)
        with self._lock:
            self._state.stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._state.frame_id = msg.header.frame_id
            self._state.color = img

    def _cb_depth(self, msg: Image) -> None:
        img = _image_to_numpy(msg)
        with self._lock:
            self._state.stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._state.frame_id = msg.header.frame_id
            self._state.depth = img

    def _cb_points(self, msg: PointCloud2) -> None:
        xyz = _pc2_to_xyz(msg)
        with self._lock:
            self._state.stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._state.frame_id = msg.header.frame_id
            self._state.pointcloud_xyz = xyz

    def get_state(self) -> CameraState:
        with self._lock:
            return self._state


def ensure_rclpy_init() -> None:
    if not rclpy.ok():
        rclpy.init(args=None)

if __name__ == "__main__":
    print("Starting Camera Node")
    rclpy.init()
    camera = CameraNode()
    camera.get_state()