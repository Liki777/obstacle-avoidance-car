"""
Gazebo + ROS2 下的简易「Gym 风格」环境（不强制依赖 gymnasium）。

前提：
- 已启动 ``ros2 launch rl_car_gazebo sim.launch.py ...``
- 已 ``source ros2_ws/install/setup.bash``（以便 ``gazebo_msgs`` 等可用）

可视化：直接使用 Gazebo GUI（本模块 ``render()`` 为空操作）。

典型流程：
    env = RlCarGazeboEnv(RobotTaskSpec.preset_diff_drive(), GazeboEnvConfig())
    s = env.reset()
    s, r, done, info = env.step(policy_action)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

from obstacle_environment.action.action_mapper import ActionMapper
from obstacle_environment.observation import ObservationConfig, build_observation
from obstacle_environment.reward import compute_reward, lidar_min_range
from obstacle_environment.robot_spec import RobotTaskSpec


def _image_msg_to_numpy(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = memoryview(msg.data)
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "bgr8":
            arr = arr[..., ::-1].copy()
        return arr
    if enc in ("mono8",):
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    return np.frombuffer(data, dtype=np.uint8)


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(math.sin(yaw * 0.5))
    q.w = float(math.cos(yaw * 0.5))
    return q


@dataclass
class GazeboEnvConfig:
    """环境与话题配置（与 ``collect_observations`` 默认一致）。"""

    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    camera_topic: str = "/depth_cam/camera/image_raw"
    goal_topic: str = "/goal_pose"
    cmd_vel_topic: str = "/cmd_vel"

    control_dt: float = 0.1
    """每步执行控制后等待的仿真/墙钟时间 (s)。"""

    max_episode_steps: int = 500
    done_on_collision: bool = True
    done_on_goal: bool = True

    # Gazebo 重置（``gazebo_msgs/SetModelState``）
    model_name: str = "rl_car"
    set_model_state_service: str = "/set_model_state"
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_z: float = 0.1
    spawn_yaw: float = 0.0
    goal_frame_id: str = "odom"

    # 固定目标；若 ``goal_sample_range`` 非空则每次 reset 在矩形内均匀随机
    goal_x: float = 2.0
    goal_y: float = 2.0
    goal_sample_range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    """例如 ((-1.0, 3.0), (-1.0, 3.0)) 表示 x,y 各自区间。"""

    reset_settle_time: float = 0.25
    """重置模型后等待传感器稳定的时间 (s)。"""

    give_success_reward: bool = True
    """是否启用到达目标的一次性奖励（见 ``RewardConfig.w_goal_success``）。"""


class _RlCarGymNode(Node):
    """订阅传感器 + 发布 ``cmd_vel`` 与 ``goal_pose``。"""

    def __init__(
        self,
        *,
        cfg: GazeboEnvConfig,
        obs_cfg: ObservationConfig,
    ) -> None:
        super().__init__("rl_car_gym_env")
        self._cfg = cfg
        self._obs_cfg = obs_cfg

        self._scan: Optional[LaserScan] = None
        self._odom: Optional[Odometry] = None
        self._image: Optional[Image] = None
        self._camera_np: Optional[np.ndarray] = None
        self._goal: Optional[PoseStamped] = None

        self._cmd_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self._goal_pub = self.create_publisher(PoseStamped, cfg.goal_topic, 10)

        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(LaserScan, cfg.scan_topic, self._on_scan, scan_qos)
        self.create_subscription(Odometry, cfg.odom_topic, self._on_odom, 10)
        if obs_cfg.include_camera:
            self.create_subscription(
                Image, cfg.camera_topic, self._on_image, qos_profile_sensor_data
            )
        self.create_subscription(PoseStamped, cfg.goal_topic, self._on_goal, 10)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_image(self, msg: Image) -> None:
        self._image = msg
        try:
            self._camera_np = _image_msg_to_numpy(msg)
        except Exception:
            self._camera_np = None

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        m = Twist()
        m.linear.x = float(linear_x)
        m.angular.z = float(angular_z)
        self._cmd_pub.publish(m)

    def stop_robot(self) -> None:
        self.publish_cmd_vel(0.0, 0.0)

    def publish_goal(self, x: float, y: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cfg.goal_frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self._goal = msg

    def ready_for_obs(self) -> bool:
        if self._scan is None or self._odom is None or self._goal is None:
            return False
        if self._obs_cfg.include_camera and self._camera_np is None:
            return False
        return True

    def build_observation(self) -> dict[str, Any]:
        if self._scan is None or self._odom is None or self._goal is None:
            raise RuntimeError("传感器或目标未就绪，请确认 Gazebo 与话题正常。")
        ranges = np.asarray(self._scan.ranges, dtype=np.float32)
        cam = self._camera_np if self._obs_cfg.include_camera else None
        gx = float(self._goal.pose.position.x)
        gy = float(self._goal.pose.position.y)
        goal_data = {"goal_xy": np.asarray([gx, gy], dtype=np.float32)}
        return build_observation(
            ranges,
            camera_data=cam,
            odom_data=self._odom,
            goal_data=goal_data,
            config=self._obs_cfg,
        )

    def get_lidar_ranges(self) -> np.ndarray:
        if self._scan is None:
            return np.asarray([], dtype=np.float32)
        return np.asarray(self._scan.ranges, dtype=np.float32)


class RlCarGazeboEnv:
    """
    ``step(action)`` → 发布速度 → 等待 ``control_dt`` → 取最新观测 → 算奖励 → 判终止。

    ``reset()`` → 停车 → ``SetModelState`` 重置位姿 → 发布目标 → 清空缓存变量 → 初始 state。
    """

    def __init__(
        self,
        spec: RobotTaskSpec,
        env_cfg: Optional[GazeboEnvConfig] = None,
    ) -> None:
        self.spec = spec
        self.env_cfg = env_cfg or GazeboEnvConfig()
        self._mapper = ActionMapper(spec.action_config)

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = _RlCarGymNode(cfg=self.env_cfg, obs_cfg=spec.observation_config)
        self._prev_goal_distance: Optional[float] = None
        self._step_idx = 0
        self._goal_reached_for_bonus = False

        self._set_model_client = None
        try:
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]

            self._set_model_client = self._node.create_client(
                SetModelState, self.env_cfg.set_model_state_service
            )
        except ImportError as e:
            self._node.get_logger().warn(
                f"未找到 gazebo_msgs，将无法重置模型位姿: {e}。请安装 ros-humble-gazebo-msgs 并在 Gazebo 中运行。"
            )

    def _pump(self, n: int = 15, timeout_sec: float = 0.02) -> None:
        for _ in range(n):
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def _wait_ready(self, timeout: float = 10.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._pump()
            if self._node.ready_for_obs():
                return
            time.sleep(0.02)
        raise TimeoutError("等待 /scan /odom /goal_pose 超时，请检查仿真是否已启动。")

    def _call_set_model_state(self) -> bool:
        if self._set_model_client is None:
            self._node.get_logger().warn("SetModelState 客户端不可用，跳过位姿重置。")
            return False
        try:
            from gazebo_msgs.msg import ModelState  # type: ignore[import-untyped]
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]
        except ImportError:
            return False

        if not self._set_model_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().warn("服务 /set_model_state 不可用，跳过位姿重置。")
            return False

        ms = ModelState()
        ms.model_name = self.env_cfg.model_name
        ms.pose.position.x = float(self.env_cfg.spawn_x)
        ms.pose.position.y = float(self.env_cfg.spawn_y)
        ms.pose.position.z = float(self.env_cfg.spawn_z)
        ms.pose.orientation = _yaw_to_quat(float(self.env_cfg.spawn_yaw))
        ms.twist = Twist()
        ms.reference_frame = "world"

        req = SetModelState.Request()
        req.model_state = ms
        fut = self._set_model_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None:
            return False
        if hasattr(res, "success") and not res.success:
            self._node.get_logger().warn(
                f"SetModelState 失败: {getattr(res, 'status_message', '')}"
            )
            return False
        return True

    def _sample_goal_xy(self) -> tuple[float, float]:
        r = self.env_cfg.goal_sample_range
        if r is None:
            return float(self.env_cfg.goal_x), float(self.env_cfg.goal_y)
        (x0, x1), (y0, y1) = r
        gx = float(np.random.uniform(min(x0, x1), max(x0, x1)))
        gy = float(np.random.uniform(min(y0, y1), max(y0, y1)))
        return gx, gy

    def reset(self) -> np.ndarray:
        """停车、重置位姿与目标、清空轨迹变量，返回初始 ``state`` 向量。"""
        self._node.stop_robot()
        self._call_set_model_state()

        gx, gy = self._sample_goal_xy()
        self._node.publish_goal(gx, gy)

        self._prev_goal_distance = None
        self._step_idx = 0
        self._goal_reached_for_bonus = False

        time.sleep(self.env_cfg.reset_settle_time)
        self._wait_ready()

        obs = self._node.build_observation()
        self._prev_goal_distance = float(obs["goal_distance"])
        return obs["state"].astype(np.float32, copy=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Returns:
            state, reward, done, info
        """
        cmd = self._mapper.to_linear_angular(action)
        self._node.publish_cmd_vel(float(cmd[0]), float(cmd[1]))

        time.sleep(self.env_cfg.control_dt)
        self._pump(20, 0.02)

        obs = self._node.build_observation()
        state = obs["state"].astype(np.float32, copy=False)
        lidar_ranges = self._node.get_lidar_ranges()
        gd = float(obs["goal_distance"])
        rcfg = self.spec.reward_config

        rb = compute_reward(
            lidar_ranges=lidar_ranges,
            linear_x=float(obs["linear_x"]),
            goal_distance=gd,
            prev_goal_distance=self._prev_goal_distance,
            config=rcfg,
            give_success_bonus=self.env_cfg.give_success_reward,
            was_goal_reached_before=self._goal_reached_for_bonus,
        )
        if self.env_cfg.give_success_reward and gd <= rcfg.goal_reached_distance:
            self._goal_reached_for_bonus = True

        self._prev_goal_distance = gd
        self._step_idx += 1

        dmin = lidar_min_range(lidar_ranges)
        terminated = False
        if self.env_cfg.done_on_collision and dmin <= rcfg.collision_distance:
            terminated = True
        if self.env_cfg.done_on_goal and gd <= rcfg.goal_reached_distance:
            terminated = True

        truncated = self._step_idx >= self.env_cfg.max_episode_steps
        done = terminated or truncated

        info = {
            "reward_breakdown": rb,
            "terminated": terminated,
            "truncated": truncated,
            "lidar_min_m": dmin,
            "goal_distance": gd,
        }
        return state, float(rb.total), done, info

    def render(self) -> None:
        """使用 Gazebo 可视化即可，此处不做额外渲染。"""
        return None

    def close(self) -> None:
        self._node.stop_robot()
        try:
            self._node.destroy_node()
        except Exception:
            pass
