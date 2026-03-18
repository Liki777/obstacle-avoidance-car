from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


@dataclass
class LidarState:
    stamp_sec: float = 0.0
    frame_id: str = ""
    angle_min: float = 0.0
    angle_max: float = 0.0
    angle_increment: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    ranges: Optional[list[float]] = None
    intensities: Optional[list[float]] = None


class LidarNode(Node):
    def __init__(self, topic: str = "/scan", node_name: str = "rl_car_lidar_bridge"):
        super().__init__(node_name)
        self._lock = threading.Lock()
        self._state = LidarState()
        self._sub = self.create_subscription(LaserScan, topic, self._cb, 10)

    def _cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._state = LidarState(
                stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                frame_id=msg.header.frame_id,
                angle_min=float(msg.angle_min),
                angle_max=float(msg.angle_max),
                angle_increment=float(msg.angle_increment),
                range_min=float(msg.range_min),
                range_max=float(msg.range_max),
                ranges=list(msg.ranges),
                intensities=list(msg.intensities) if msg.intensities else None,
            )

    def get_state(self) -> LidarState:
        with self._lock:
            return self._state


def ensure_rclpy_init() -> None:
    if not rclpy.ok():
        rclpy.init(args=None)

if __name__ == "__main__":
    print("Starting Lidar Node")
    rclpy.init()
    lidar = LidarNode()
    lidar.get_state()