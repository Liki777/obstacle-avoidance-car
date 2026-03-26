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
import csv
import os
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
    done_on_out_of_bounds: bool = True
    # 基于 maze.world（Maze-Map）外墙中心点推断的训练边界（单位：m，world 坐标）
    # 墙中心大致落在 x∈[0,10], y∈[-1,9]，这里加一点裕量避免贴墙误判
    map_x_min: float = -0.5
    map_x_max: float = 10.5
    map_y_min: float = -1.5
    map_y_max: float = 9.5

    # Gazebo 重置（``gazebo_msgs/SetModelState``）
    model_name: str = "rl_car"
    # 不同发行版/启动方式下服务名可能不同；按顺序尝试，并在等待期间 spin 以发现服务
    set_model_state_services: tuple[str, ...] = (
        "/set_model_state",
        "/gazebo/set_model_state",
    )
    set_entity_state_services: tuple[str, ...] = (
        "/set_entity_state",
        "/gazebo/set_entity_state",
    )
    set_model_state_wait_sec: float = 2.0
    """为「每个候选服务名」分配的最大等待时间（秒）；多候选时会依次尝试。"""
    reset_world_service: str = "/reset_world"
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
    goal_min_distance: float = 0.8
    """若启用 goal_sample_range，采样目标与出生点的最小距离（m），避免目标太近无学习信号。"""

    reset_settle_time: float = 0.25
    """重置模型后等待传感器稳定的时间 (s)。"""

    give_success_reward: bool = True
    """是否启用到达目标的一次性奖励（见 ``RewardConfig.w_goal_success``）。"""

    # ---- 日志：每一步写入 CSV（包含 action + 观测关键信息 + reward 分项）----
    step_log_csv: Optional[str] = None
    """例如 'logs/ppo_steps.csv'。None 表示不记录。"""


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
        self._prev_action: Optional[np.ndarray] = None
        self._episode_idx = 0

        # ---- step 日志（可选）----
        self._step_log_f = None
        self._step_log_writer = None
        if self.env_cfg.step_log_csv:
            path = self.env_cfg.step_log_csv
            if not os.path.isabs(path):
                # 相对路径按当前工作目录解析（通常是项目根）
                path = os.path.abspath(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._step_log_f = open(path, "w", newline="", encoding="utf-8")
            self._step_log_writer = csv.writer(self._step_log_f)
            # 头：包含 action、lidar 下采样（obs['lidar']）、关键观测与 reward 分项
            header = [
                "t_wall",
                "episode",
                "step",
                "raw_action_0",
                "raw_action_1",
                "cmd_linear_x",
                "cmd_angular_z",
                "pos_x",
                "pos_y",
                "goal_distance",
                "goal_angle",
                "linear_x",
                "angular_z",
                "lidar_min",
                "done",
                "terminated",
                "truncated",
                "terminal_reason",
                # reward breakdown (flatten)
                "reward_total",
                "reward_progress",
                "reward_safe",
                "reward_risk",
                "reward_turn",
                "reward_stop",
                "reward_smooth",
                "reward_terminal",
            ] + [f"lidar_feat_{i}" for i in range(int(self.spec.observation_config.lidar_dim))]
            self._step_log_writer.writerow(header)
            self._step_log_f.flush()

        self._set_model_client: Any = None
        self._set_entity_client: Any = None
        self._reset_world_client: Any = None
        self._gazebo_msgs_ok = False
        self._set_model_resolve_warned = False
        self._disable_set_model_resolution = False
        self._disable_set_entity_resolution = False
        try:
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]  # noqa: F401

            self._gazebo_msgs_ok = True
        except ImportError as e:
            self._node.get_logger().warn(
                f"未找到 gazebo_msgs，将无法重置模型位姿: {e}。请安装 ros-humble-gazebo-msgs 并在 Gazebo 中运行。"
            )
        try:
            from std_srvs.srv import Empty  # type: ignore[import-untyped]

            self._reset_world_client = self._node.create_client(
                Empty, self.env_cfg.reset_world_service
            )
        except Exception:
            self._reset_world_client = None

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

    def _discover_set_model_state_service_names(self) -> list[str]:
        """从当前图中查找类型为 gazebo_msgs/srv/SetModelState 的服务名（含命名空间）。"""
        target = "gazebo_msgs/srv/SetModelState"
        found: list[str] = []
        try:
            for name, types in self._node.get_service_names_and_types():
                if target in types:
                    found.append(name)
        except Exception:
            pass
        return sorted(found)

    def _discover_set_entity_state_service_names(self) -> list[str]:
        """从当前图中查找类型为 gazebo_msgs/srv/SetEntityState 的服务名。"""
        target = "gazebo_msgs/srv/SetEntityState"
        found: list[str] = []
        try:
            for name, types in self._node.get_service_names_and_types():
                if target in types:
                    found.append(name)
        except Exception:
            pass
        return sorted(found)

    def _resolve_set_model_client(self) -> bool:
        """懒连接：边 spin 边等；先 ROS2 图自动发现 SetModelState，再试配置里的候选名。"""
        if self._disable_set_model_resolution:
            return False
        if self._set_model_client is not None:
            return True
        if not self._gazebo_msgs_ok:
            return False
        try:
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]
        except ImportError:
            return False

        # 先 pump，便于 get_service_names_and_types 与 service_is_ready 更新
        self._pump(20, 0.02)
        discovered = self._discover_set_model_state_service_names()
        # 去重且保持顺序：自动发现优先，其次配置文件中的固定名
        seen: set[str] = set()
        candidates: list[str] = []
        for s in discovered + list(self.env_cfg.set_model_state_services):
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        if len(discovered) == 0:
            # 图中完全没有该类型服务：当前运行通常不支持，直接禁用后续重复探测
            self._disable_set_model_resolution = True
            return False

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(SetModelState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._set_model_client = cli
                    self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（SetModelState）")
                    return True
                self._pump(15, 0.02)
            try:
                self._node.destroy_client(cli)
            except Exception:
                pass

        if not self._set_model_resolve_warned:
            self._set_model_resolve_warned = True
            self._node.get_logger().warn(
                "未发现类型为 gazebo_msgs/srv/SetModelState 的服务（已尝试自动发现 + "
                f"{list(self.env_cfg.set_model_state_services)}）。"
                " 若只有 /get_model_list：请执行 `ros2 service list | grep -i set` 或 `ros2 service list | grep gazebo`，"
                " 确认 gzserver 是否提供 SetModelState；Classic 下需 gazebo_ros 正常加载。"
                " 将跳过位姿重置（小车从当前位置继续训练）。"
            )
        self._disable_set_model_resolution = True
        return False

    def _resolve_set_entity_client(self) -> bool:
        if self._disable_set_entity_resolution:
            return False
        if self._set_entity_client is not None:
            return True
        if not self._gazebo_msgs_ok:
            return False
        try:
            from gazebo_msgs.srv import SetEntityState  # type: ignore[import-untyped]
        except ImportError:
            return False

        self._pump(20, 0.02)
        discovered = self._discover_set_entity_state_service_names()
        seen: set[str] = set()
        candidates: list[str] = []
        for s in discovered + list(self.env_cfg.set_entity_state_services):
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        if len(discovered) == 0:
            self._disable_set_entity_resolution = True
            return False

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(SetEntityState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._set_entity_client = cli
                    self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（SetEntityState）")
                    return True
                self._pump(15, 0.02)
            try:
                self._node.destroy_client(cli)
            except Exception:
                pass
        self._disable_set_entity_resolution = True
        return False

    def _call_set_model_state(self) -> bool:
        if not self._resolve_set_model_client():
            return False
        try:
            from gazebo_msgs.msg import ModelState  # type: ignore[import-untyped]
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]
        except ImportError:
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

    def _call_set_entity_state(self) -> bool:
        if not self._resolve_set_entity_client():
            return False
        try:
            from gazebo_msgs.msg import EntityState  # type: ignore[import-untyped]
            from gazebo_msgs.srv import SetEntityState  # type: ignore[import-untyped]
        except ImportError:
            return False

        st = EntityState()
        st.name = self.env_cfg.model_name
        st.pose.position.x = float(self.env_cfg.spawn_x)
        st.pose.position.y = float(self.env_cfg.spawn_y)
        st.pose.position.z = float(self.env_cfg.spawn_z)
        st.pose.orientation = _yaw_to_quat(float(self.env_cfg.spawn_yaw))
        st.twist = Twist()
        st.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = st
        fut = self._set_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None:
            return False
        if hasattr(res, "success") and not res.success:
            self._node.get_logger().warn(
                f"SetEntityState 失败: {getattr(res, 'status_message', '')}"
            )
            return False
        return True

    def _sample_goal_xy(self) -> tuple[float, float]:
        r = self.env_cfg.goal_sample_range
        if r is None:
            return float(self.env_cfg.goal_x), float(self.env_cfg.goal_y)
        (x0, x1), (y0, y1) = r
        # 采样直到与 spawn 足够远
        for _ in range(200):
            gx = float(np.random.uniform(min(x0, x1), max(x0, x1)))
            gy = float(np.random.uniform(min(y0, y1), max(y0, y1)))
            dx = gx - float(self.env_cfg.spawn_x)
            dy = gy - float(self.env_cfg.spawn_y)
            if float(np.hypot(dx, dy)) >= float(self.env_cfg.goal_min_distance):
                return gx, gy
        return gx, gy

    @staticmethod
    def _extract_position_xy_from_obs(obs: dict[str, Any]) -> tuple[float, float] | None:
        odom = obs.get("odom")
        if odom is None:
            return None
        try:
            p = odom.pose.pose.position
            return float(p.x), float(p.y)
        except Exception:
            return None

    def _is_out_of_bounds(self, pos_xy: tuple[float, float] | None) -> bool:
        if pos_xy is None:
            return False
        x, y = pos_xy
        return (
            x < float(self.env_cfg.map_x_min)
            or x > float(self.env_cfg.map_x_max)
            or y < float(self.env_cfg.map_y_min)
            or y > float(self.env_cfg.map_y_max)
        )

    def reset(self) -> np.ndarray:
        """停车、重置位姿与目标、清空轨迹变量，返回初始 ``state`` 向量。"""
        self._node.stop_robot()
        # 优先 set_model_state；失败时退化到 /reset_world，避免一直从错误姿态继续跑飞
        ok = self._call_set_model_state()
        if not ok:
            ok = self._call_set_entity_state()
        if (not ok) and (self._reset_world_client is not None):
            try:
                from std_srvs.srv import Empty  # type: ignore[import-untyped]

                if self._reset_world_client.wait_for_service(timeout_sec=2.0):
                    fut = self._reset_world_client.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(self._node, fut, timeout_sec=3.0)
                    ok = True
            except Exception:
                pass
        if (not ok) and (not self._set_model_resolve_warned):
            self._set_model_resolve_warned = True
            self._node.get_logger().warn(
                "未发现 set_model_state/set_entity_state，且 /reset_world 不可用；重置将仅发布零速度并继续。"
            )

        gx, gy = self._sample_goal_xy()
        self._node.publish_goal(gx, gy)

        self._prev_goal_distance = None
        self._step_idx = 0
        self._goal_reached_for_bonus = False
        self._prev_action = None
        self._episode_idx += 1

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
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        a0 = float(raw_action[0]) if raw_action.size > 0 else 0.0
        a1 = float(raw_action[1]) if raw_action.size > 1 else 0.0

        cmd = self._mapper.to_linear_angular(raw_action)
        self._node.publish_cmd_vel(float(cmd[0]), float(cmd[1]))

        time.sleep(self.env_cfg.control_dt)
        self._pump(20, 0.02)

        obs = self._node.build_observation()
        state = obs["state"].astype(np.float32, copy=False)
        lidar_ranges = self._node.get_lidar_ranges()
        gd = float(obs["goal_distance"])
        rcfg = self.spec.reward_config

        dmin = lidar_min_range(lidar_ranges)
        pos_xy = self._extract_position_xy_from_obs(obs)
        out_of_bounds = self._is_out_of_bounds(pos_xy)
        terminated = False
        terminal_reason: Optional[str] = None
        if self.env_cfg.done_on_collision and dmin <= rcfg.collision_distance:
            terminated = True
            terminal_reason = "collision"
        if (not terminated) and self.env_cfg.done_on_goal and gd <= rcfg.goal_reached_distance:
            terminated = True
            terminal_reason = "success"
        if (not terminated) and self.env_cfg.done_on_out_of_bounds and out_of_bounds:
            terminated = True
            terminal_reason = "out_of_bounds"

        rb = compute_reward(
            lidar_ranges=lidar_ranges,
            linear_x=float(obs["linear_x"]),
            angular_z=float(obs.get("angular_z", 0.0)),
            goal_distance=gd,
            goal_angle=float(obs.get("goal_angle", 0.0)),
            prev_goal_distance=self._prev_goal_distance,
            action=cmd,
            prev_action=self._prev_action,
            config=rcfg,
            terminal=terminal_reason,
        )

        self._prev_action = cmd.astype(np.float32, copy=False)
        self._prev_goal_distance = gd
        self._step_idx += 1

        truncated = self._step_idx >= self.env_cfg.max_episode_steps
        done = terminated or truncated

        info = {
            "reward_breakdown": rb,
            "terminated": terminated,
            "truncated": truncated,
            "lidar_min_m": dmin,
            "goal_distance": gd,
            "position_xy": pos_xy,
            "out_of_bounds": out_of_bounds,
            "terminal_reason": terminal_reason or "",
        }

        # ---- 写 step 日志（可选）----
        if self._step_log_writer is not None and self._step_log_f is not None:
            pos_x = float(pos_xy[0]) if pos_xy is not None else 0.0
            pos_y = float(pos_xy[1]) if pos_xy is not None else 0.0
            lidar_feat = obs.get("lidar")
            if isinstance(lidar_feat, np.ndarray):
                lf = lidar_feat.reshape(-1).astype(np.float32, copy=False)
            else:
                lf = np.zeros((int(self.spec.observation_config.lidar_dim),), dtype=np.float32)
            lf = lf[: int(self.spec.observation_config.lidar_dim)]
            if lf.size < int(self.spec.observation_config.lidar_dim):
                lf = np.pad(lf, (0, int(self.spec.observation_config.lidar_dim) - lf.size))

            rb_dict = rb.as_dict()
            row = [
                f"{time.time():.6f}",
                str(self._episode_idx),
                str(self._step_idx),
                f"{a0:.6f}",
                f"{a1:.6f}",
                f"{float(cmd[0]):.6f}",
                f"{float(cmd[1]):.6f}",
                f"{pos_x:.6f}",
                f"{pos_y:.6f}",
                f"{gd:.6f}",
                f"{float(obs.get('goal_angle', 0.0)):.6f}",
                f"{float(obs.get('linear_x', 0.0)):.6f}",
                f"{float(obs.get('angular_z', 0.0)):.6f}",
                f"{float(dmin):.6f}",
                "1" if done else "0",
                "1" if terminated else "0",
                "1" if truncated else "0",
                terminal_reason or "",
                f"{rb_dict.get('reward_total', 0.0):.6f}",
                f"{rb_dict.get('reward_progress', 0.0):.6f}",
                f"{rb_dict.get('reward_safe', 0.0):.6f}",
                f"{rb_dict.get('reward_risk', 0.0):.6f}",
                f"{rb_dict.get('reward_turn', 0.0):.6f}",
                f"{rb_dict.get('reward_stop', 0.0):.6f}",
                f"{rb_dict.get('reward_smooth', 0.0):.6f}",
                f"{rb_dict.get('reward_terminal', 0.0):.6f}",
            ] + [f"{float(x):.6f}" for x in lf.tolist()]
            self._step_log_writer.writerow(row)
            # 为了可实时观察，定期 flush（每 10 步）
            if (self._step_idx % 10) == 0:
                try:
                    self._step_log_f.flush()
                except Exception:
                    pass

        return state, float(rb.total), done, info

    def render(self) -> None:
        """使用 Gazebo 可视化即可，此处不做额外渲染。"""
        return None

    def close(self) -> None:
        self._node.stop_robot()
        if self._step_log_f is not None:
            try:
                self._step_log_f.flush()
            except Exception:
                pass
            try:
                self._step_log_f.close()
            except Exception:
                pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
