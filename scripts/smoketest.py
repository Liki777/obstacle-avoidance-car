from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, LaserScan


@dataclass
class Stats:
    count: int = 0
    last_t_wall: float = 0.0
    hz_ema: float = 0.0

    def tick(self) -> None:
        now = time.time()
        if self.last_t_wall > 0:
            dt = now - self.last_t_wall
            if dt > 0:
                hz = 1.0 / dt
                self.hz_ema = hz if self.hz_ema == 0.0 else (0.9 * self.hz_ema + 0.1 * hz)
        self.last_t_wall = now
        self.count += 1


class SmokeTestNode(Node):
    def __init__(self):
        super().__init__("rl_car_smoke_test")

        # Topics based on what Gazebo is actually publishing in your setup.
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.odom_topic = self.declare_parameter("odom_topic", "/odom").value
        self.rgb_topic = self.declare_parameter("rgb_topic", "/depth_cam/camera/image_raw").value
        self.depth_topic = self.declare_parameter("depth_topic", "/depth_cam/camera/depth/image_raw").value
        self.cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel").value

        self._pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self._scan_stats = Stats()
        self._odom_stats = Stats()
        self._rgb_stats = Stats()
        self._depth_stats = Stats()

        self._last_scan_stamp: Optional[float] = None
        self._last_odom_xy: Optional[tuple[float, float]] = None

        # Match common Gazebo QoS:
        # - /scan in this setup is RELIABLE (see `ros2 topic info -v /scan`)
        # - camera topics are usually BEST_EFFORT (SensorDataQoS)
        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = QoSReliabilityPolicy.RELIABLE

        cam_qos: QoSProfile = qos_profile_sensor_data
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, scan_qos)
        self.create_subscription(Image, self.rgb_topic, self._on_rgb, cam_qos)
        self.create_subscription(Image, self.depth_topic, self._on_depth, cam_qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)

        self._t0 = time.time()
        self._phase = 0

        self.create_timer(0.1, self._publish_cmd)
        self.create_timer(1.0, self._print_status)

        self.get_logger().info(
            "Smoke test started.\n"
            f"  scan:  {self.scan_topic}\n"
            f"  odom:  {self.odom_topic}\n"
            f"  rgb:   {self.rgb_topic}\n"
            f"  depth: {self.depth_topic}\n"
            f"  cmd:   {self.cmd_vel_topic}\n"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_stats.tick()
        self._last_scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_stats.tick()
        p = msg.pose.pose.position
        self._last_odom_xy = (float(p.x), float(p.y))

    def _on_rgb(self, msg: Image) -> None:
        self._rgb_stats.tick()

    def _on_depth(self, msg: Image) -> None:
        self._depth_stats.tick()

    def _publish_cmd(self) -> None:
        # 0-5s: slow forward, 5-10s: rotate, then repeat.
        t = time.time() - self._t0
        phase = int(t // 5) % 2
        if phase != self._phase:
            self._phase = phase
            self.get_logger().info(f"Switching motion phase -> {self._phase}")

        msg = Twist()
        if self._phase == 0:
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.6

        # small sinusoidal wobble to ensure non-constant command
        msg.angular.z += 0.1 * math.sin(t)
        self._pub.publish(msg)

    def _print_status(self) -> None:
        scan = f"{self._scan_stats.hz_ema:5.1f}Hz ({self._scan_stats.count})"
        odom = f"{self._odom_stats.hz_ema:5.1f}Hz ({self._odom_stats.count})"
        rgb = f"{self._rgb_stats.hz_ema:5.1f}Hz ({self._rgb_stats.count})"
        dep = f"{self._depth_stats.hz_ema:5.1f}Hz ({self._depth_stats.count})"

        extra = []
        if self._last_scan_stamp is not None:
            extra.append(f"scan_stamp={self._last_scan_stamp:.3f}")
        if self._last_odom_xy is not None:
            extra.append(f"odom_xy=({self._last_odom_xy[0]:.3f},{self._last_odom_xy[1]:.3f})")

        self.get_logger().info(f"scan {scan} | odom {odom} | rgb {rgb} | depth {dep} | " + " ".join(extra))


def main() -> None:
    rclpy.init()
    node = SmokeTestNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

