from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


def ensure_rclpy_init() -> None:
    if not rclpy.ok():
        rclpy.init(args=None)


@dataclass
class Action:
    linear_x: float
    angular_z: float


class _CmdVelNode(Node):
    def __init__(self, topic: str = "/cmd_vel", node_name: str = "rl_car_control_bridge"):
        super().__init__(node_name)
        self._pub = self.create_publisher(Twist, topic, 10)
        self._lock = threading.Lock()
        self._last: Optional[Action] = None

    def publish(self, action: Action) -> None:
        msg = Twist()
        msg.linear.x = float(action.linear_x)
        msg.angular.z = float(action.angular_z)
        self._pub.publish(msg)
        with self._lock:
            self._last = action

    def stop(self) -> None:
        self.publish(Action(0.0, 0.0))

    def last_action(self) -> Optional[Action]:
        with self._lock:
            return self._last


class ControlBridge:
    """
    统一控制接口（对应 gazebo_ros_diff_drive）：
    - 默认订阅 /cmd_vel
    """

    def __init__(self, cmd_vel_topic: str = "/cmd_vel", spin_in_background: bool = True):
        ensure_rclpy_init()
        self._node = _CmdVelNode(topic=cmd_vel_topic)
        self._executor = MultiThreadedExecutor(num_threads=1)
        self._executor.add_node(self._node)

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
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass

    def set_action(self, linear_x: float, angular_z: float) -> None:
        self._node.publish(Action(linear_x=linear_x, angular_z=angular_z))

    def set_action_from_array(self, action: Sequence[float]) -> None:
        if len(action) < 2:
            raise ValueError("action 至少需要 [linear_x, angular_z]")
        self.set_action(float(action[0]), float(action[1]))

    def stop(self) -> None:
        self._node.stop()

    def last_action(self) -> Optional[Action]:
        return self._node.last_action()

if __name__ == "__main__":
    print("Starting Control Node")
    rclpy.init()
    control = ControlBridge()
    control.set_action(1.0, 1.0)
    control.stop()
    control.last_action()