#!/usr/bin/env python3
"""
验证脚本（不涉及任何模型/pt）：

1) 持续发布 /cmd_vel（默认直行前进），用于确认小车“正面”方向
2) 订阅 /scan 与深度相机话题，周期性打印：
   - 雷达最小距离 d_min
   - 深度图中心像素 depth_center 与有限像素均值 depth_mean

配合 Gazebo GUI 使用即可完成可视化验证。
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = memoryview(msg.data)
    if enc in ("32fc1",):
        return np.frombuffer(data, dtype=np.float32).reshape((h, w))
    if enc in ("16uc1",):
        return np.frombuffer(data, dtype=np.uint16).reshape((h, w))
    if enc in ("mono8",):
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    # fallback（不保证 reshape）
    return np.frombuffer(data, dtype=np.uint8)


class DriveAndCheck(Node):
    def __init__(
        self,
        *,
        cmd_vel_topic: str,
        scan_topic: str,
        depth_topic: str,
        linear_x: float,
        angular_z: float,
        pub_hz: float,
        print_every_sec: float,
    ) -> None:
        super().__init__("rl_car_drive_forward_and_check")
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(LaserScan, scan_topic, self._on_scan, scan_qos)
        self.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)

        self._linear_x = float(linear_x)
        self._angular_z = float(angular_z)

        self._last_scan_t: Optional[float] = None
        self._last_depth_t: Optional[float] = None
        self._scan_dmin: Optional[float] = None
        self._depth_center: Optional[float] = None
        self._depth_mean: Optional[float] = None
        self._depth_hw: tuple[int, int] = (0, 0)

        self._last_print_wall = time.time()
        self._print_every_sec = float(print_every_sec)

        period = 1.0 / max(float(pub_hz), 1e-3)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            "Drive+Check started.\n"
            f"  cmd:   {cmd_vel_topic}\n"
            f"  scan:  {scan_topic}\n"
            f"  depth: {depth_topic}\n"
            f"  v={self._linear_x} w={self._angular_z} pub_hz={pub_hz}\n"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        r = np.asarray(msg.ranges, dtype=np.float32).reshape(-1)
        finite = r[np.isfinite(r)]
        self._scan_dmin = float(np.min(finite)) if finite.size else None
        self._last_scan_t = time.time()

    def _on_depth(self, msg: Image) -> None:
        arr = image_msg_to_numpy(msg)
        if arr.ndim == 2:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            self._depth_hw = (h, w)
            cy, cx = h // 2, w // 2
            center = float(arr[cy, cx])
            # depth 常见：0 或 inf 表示无效；这里统计有限且 >0 的均值
            a = np.asarray(arr, dtype=np.float32)
            finite = a[np.isfinite(a) & (a > 0)]
            self._depth_center = center
            self._depth_mean = float(np.mean(finite)) if finite.size else None
        self._last_depth_t = time.time()

    def _tick(self) -> None:
        # publish cmd_vel
        msg = Twist()
        msg.linear.x = float(self._linear_x)
        msg.angular.z = float(self._angular_z)
        self._cmd_pub.publish(msg)

        now = time.time()
        if now - self._last_print_wall < self._print_every_sec:
            return
        self._last_print_wall = now

        scan_age = None if self._last_scan_t is None else (now - self._last_scan_t)
        depth_age = None if self._last_depth_t is None else (now - self._last_depth_t)

        h, w = self._depth_hw
        self.get_logger().info(
            "status | "
            f"scan_age={scan_age if scan_age is not None else -1:.2f}s "
            f"dmin={self._scan_dmin if self._scan_dmin is not None else float('nan'):.3f} | "
            f"depth_age={depth_age if depth_age is not None else -1:.2f}s "
            f"hw={h}x{w} "
            f"center={self._depth_center if self._depth_center is not None else float('nan'):.3f} "
            f"mean={self._depth_mean if self._depth_mean is not None else float('nan'):.3f}"
        )

    def stop(self) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self._cmd_pub.publish(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-vel-topic", type=str, default="/cmd_vel")
    ap.add_argument("--scan-topic", type=str, default="/scan")
    ap.add_argument("--depth-topic", type=str, default="/depth_cam/camera/depth/image_raw")
    ap.add_argument("--linear-x", type=float, default=0.4)
    ap.add_argument("--angular-z", type=float, default=0.0)
    ap.add_argument("--pub-hz", type=float, default=10.0)
    ap.add_argument("--print-every-sec", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=0.0, help="0=一直跑，>0 运行多少秒后退出")
    args = ap.parse_args()

    rclpy.init()
    node = DriveAndCheck(
        cmd_vel_topic=args.cmd_vel_topic,
        scan_topic=args.scan_topic,
        depth_topic=args.depth_topic,
        linear_x=args.linear_x,
        angular_z=args.angular_z,
        pub_hz=args.pub_hz,
        print_every_sec=args.print_every_sec,
    )
    try:
        if float(args.duration) > 0:
            end_t = time.time() + float(args.duration)
            while rclpy.ok() and time.time() < end_t:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

