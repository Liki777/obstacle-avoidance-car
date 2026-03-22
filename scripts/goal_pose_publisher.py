#!/usr/bin/env python3
"""
持续发布 /goal_pose (geometry_msgs/PoseStamped)，供 collect_observations 等节点订阅。
坐标在 map/odom 中语义由你 launch 决定；默认 frame_id=odom。
"""
from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class GoalPosePublisher(Node):
    def __init__(self, *, frame_id: str, x: float, y: float, rate_hz: float) -> None:
        super().__init__("rl_car_goal_pose_publisher")
        self._pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._frame_id = frame_id
        self._x = x
        self._y = y
        period = 1.0 / max(rate_hz, 0.1)
        self.create_timer(period, self._publish)

    def _publish(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(self._x)
        msg.pose.position.y = float(self._y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._pub.publish(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=2.0)
    ap.add_argument("--y", type=float, default=2.0)
    ap.add_argument("--frame-id", type=str, default="odom")
    ap.add_argument("--rate", type=float, default=5.0, help="发布频率 Hz")
    args = ap.parse_args()

    rclpy.init()
    node = GoalPosePublisher(frame_id=args.frame_id, x=args.x, y=args.y, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
