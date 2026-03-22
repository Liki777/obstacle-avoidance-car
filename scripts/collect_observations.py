#!/usr/bin/env python3
"""
从 ROS2 话题订阅 lidar / camera / odom / goal，调用 observation_builder.build_observation，
将 state 与元数据写入 CSV。

无任何随机数：四类数据均来自订阅回调里缓存的最新消息。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from obstacle_environment.observation import ObservationConfig, build_observation


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = memoryview(msg.data)
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = arr[..., ::-1].copy()
        return arr
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))[..., :3]
        if enc == "bgra8":
            arr = arr[..., ::-1].copy()
        return arr
    if enc == "mono8":
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    if enc == "16uc1":
        return np.frombuffer(data, dtype=np.uint16).reshape((h, w))
    if enc == "32fc1":
        return np.frombuffer(data, dtype=np.float32).reshape((h, w))
    return np.frombuffer(data, dtype=np.uint8)


class ObservationCollector(Node):
    def __init__(
        self,
        *,
        scan_topic: str,
        odom_topic: str,
        camera_topic: str,
        goal_topic: str,
        lidar_dim: int,
        lidar_reduce: str,
        include_camera_in_state: bool,
        camera_feature_dim: int,
        sample_hz: float,
        output_csv: str,
        require_camera: bool,
    ) -> None:
        super().__init__("rl_car_observation_collector")

        self._scan_msg: Optional[LaserScan] = None
        self._odom_msg: Optional[Odometry] = None
        self._image_msg: Optional[Image] = None
        self._goal_msg: Optional[PoseStamped] = None
        self._camera_np: Optional[np.ndarray] = None

        self._cfg = ObservationConfig(
            lidar_dim=lidar_dim,
            lidar_reduce=lidar_reduce,
            include_camera=include_camera_in_state,
            camera_feature_dim=camera_feature_dim,
        )
        self._state_dim = self._cfg.state_dim()

        self._require_camera = require_camera
        self._sample_period = 1.0 / float(sample_hz)
        self._last_print = time.time()
        self._count = 0

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        self._f = open(output_csv, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)

        header = [
            "t_wall",
            "scan_stamp_sec",
            "odom_stamp_sec",
            "camera_stamp_sec",
            "goal_stamp_sec",
            "goal_x",
            "goal_y",
            "camera_h",
            "camera_w",
            "velocity",
            "goal_distance",
            "goal_angle",
        ] + [f"state_{i}" for i in range(self._state_dim)]
        self._writer.writerow(header)
        self._f.flush()

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = QoSReliabilityPolicy.RELIABLE
        cam_qos = qos_profile_sensor_data

        self.create_subscription(LaserScan, scan_topic, self._on_scan, scan_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Image, camera_topic, self._on_image, cam_qos)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)

        self.get_logger().info(
            f"Subscribed: scan={scan_topic} odom={odom_topic} camera={camera_topic} goal={goal_topic}"
        )
        self.create_timer(self._sample_period, self._tick)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_msg = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_msg = msg

    def _on_image(self, msg: Image) -> None:
        self._image_msg = msg
        try:
            self._camera_np = image_msg_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f"camera decode failed: {e}")
            self._camera_np = None

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_msg = msg

    def _goal_dict(self) -> Optional[dict[str, Any]]:
        if self._goal_msg is None:
            return None
        p = self._goal_msg.pose.position
        return {"goal_xy": np.asarray([float(p.x), float(p.y)], dtype=np.float32)}

    def _tick(self) -> None:
        if self._scan_msg is None or self._odom_msg is None or self._goal_msg is None:
            return
        if self._require_camera and (self._camera_np is None):
            return

        scan = self._scan_msg
        odom = self._odom_msg
        goal_data = self._goal_dict()
        assert goal_data is not None

        lidar_ranges = np.asarray(scan.ranges, dtype=np.float32)
        camera_data = self._camera_np  # numpy from subscribed Image; may be None if no frame yet and not require_camera

        obs = build_observation(
            lidar_ranges,
            camera_data=camera_data,
            odom_data=odom,
            goal_data=goal_data,
            config=self._cfg,
        )

        state = obs["state"].astype(np.float32, copy=False)
        velocity = float(obs["velocity"])
        goal_distance = float(obs["goal_distance"])
        goal_angle = float(obs["goal_angle"])

        scan_stamp = float(scan.header.stamp.sec) + float(scan.header.stamp.nanosec) * 1e-9
        odom_stamp = float(odom.header.stamp.sec) + float(odom.header.stamp.nanosec) * 1e-9
        cam_stamp = (
            float(self._image_msg.header.stamp.sec) + float(self._image_msg.header.stamp.nanosec) * 1e-9
            if self._image_msg is not None
            else -1.0
        )
        goal_stamp = float(self._goal_msg.header.stamp.sec) + float(self._goal_msg.header.stamp.nanosec) * 1e-9
        gx = float(self._goal_msg.pose.position.x)
        gy = float(self._goal_msg.pose.position.y)
        ch = int(self._camera_np.shape[0]) if self._camera_np is not None else 0
        cw = int(self._camera_np.shape[1]) if self._camera_np is not None else 0

        t_wall = time.time()
        row = [
            f"{t_wall:.6f}",
            f"{scan_stamp:.6f}",
            f"{odom_stamp:.6f}",
            f"{cam_stamp:.6f}",
            f"{goal_stamp:.6f}",
            f"{gx:.6f}",
            f"{gy:.6f}",
            f"{ch}",
            f"{cw}",
            f"{velocity:.6f}",
            f"{goal_distance:.6f}",
            f"{goal_angle:.6f}",
        ] + [f"{x:.6f}" for x in state.tolist()]
        self._writer.writerow(row)
        self._count += 1
        if self._count % 10 == 0:
            self._f.flush()

        now = time.time()
        if now - self._last_print > 1.0:
            self.get_logger().info(
                f"obs saved: {self._count} | v={velocity:.3f} goal_d={goal_distance:.3f} cam={ch}x{cw}"
            )
            self._last_print = now

    def close(self) -> None:
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Subscribe lidar/camera/odom/goal, log build_observation state to CSV.")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--scan-topic", type=str, default="/scan")
    ap.add_argument("--odom-topic", type=str, default="/odom")
    ap.add_argument("--camera-topic", type=str, default="/depth_cam/camera/image_raw")
    ap.add_argument("--goal-topic", type=str, default="/goal_pose")
    ap.add_argument("--lidar-dim", type=int, default=15)
    ap.add_argument("--lidar-reduce", type=str, default="min", choices=["min", "mean"])
    ap.add_argument(
        "--include-camera-in-state",
        action="store_true",
        help="若为真，state 末尾拼接 camera_feature_dim 维占位（需 CNN 时再改 camera_processor）",
    )
    ap.add_argument("--camera-feature-dim", type=int, default=0)
    ap.add_argument(
        "--no-require-camera",
        action="store_true",
        help="未收到相机也写行（camera 时间戳为 -1，h/w 为 0）",
    )
    ap.add_argument("--output", type=str, default="/tmp/rl_car_obs.csv")
    args = ap.parse_args()

    require_cam = not args.no_require_camera

    rclpy.init()
    node = ObservationCollector(
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        camera_topic=args.camera_topic,
        goal_topic=args.goal_topic,
        lidar_dim=args.lidar_dim,
        lidar_reduce=args.lidar_reduce,
        include_camera_in_state=args.include_camera_in_state,
        camera_feature_dim=max(0, args.camera_feature_dim),
        sample_hz=args.sample_hz,
        output_csv=args.output,
        require_camera=require_cam,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    end_t = time.time() + float(args.duration)
    try:
        while rclpy.ok() and time.time() < end_t:
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
