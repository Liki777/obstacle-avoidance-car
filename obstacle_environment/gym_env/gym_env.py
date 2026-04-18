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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

from obstacle_environment.action.action_mapper import ActionMapper
from obstacle_environment.observation import ObservationConfig, build_observation
from obstacle_environment.reward import compute_reward, lidar_front_min_range, lidar_min_range
from obstacle_environment.robot_spec import RobotTaskSpec
from obstacle_environment.scenario_manager import GazeboObstacleManager, GazeboObstacleManagerConfig
from obstacle_environment.world_generator import StaticObstacleSamplingConfig, sample_static_obstacles


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


def _linear_cmd_sigmoid_scale(d_front: float, *, k: float, mid_m: float) -> float:
    """
    σ(k·(d - mid))：前向净空大 → 接近 1；很近 → 接近 0，用于缩放 cmd.linear.x。
    """
    if not math.isfinite(d_front) or d_front <= 0.0:
        return 0.02
    x = float(k) * (float(d_front) - float(mid_m))
    x = max(-60.0, min(60.0, x))
    return float(1.0 / (1.0 + math.exp(-x)))


def _lidar_has_reliable_close_reading(
    lidar_ranges: np.ndarray,
    *,
    range_max: float | None,
    collision_distance: float,
) -> bool:
    """
    判断当前 LaserScan 是否“足够可信”用于碰撞终止。

    背景：``lidar_min_range()`` 在「无有效回波/全 inf」时会返回 0.0，
    若直接拿 ``dmin <= collision_distance`` 做 done，会在第 0 步误触发 collision（你看到的 dim=0.0）。
    """
    r = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
    if r.size < 8:
        return False

    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return False

    # 若提供了 range_max：要求存在明显“非满量程”的近距离回波，才认为激光有效
    if range_max is not None:
        hi = float(range_max)
        eps = max(5e-4, 0.002 * hi)
        thr = hi - eps
        close = finite[finite < thr]
        if close.size < 8:
            return False
        dmin = float(np.min(close))
        return dmin <= float(collision_distance) + 1e-3

    # 无 range_max：退化为“有足够多的有限读数”
    return int(np.sum(np.isfinite(r))) >= max(8, int(0.2 * r.size))


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

    # out_of_bounds 判定用的位姿来源：
    # - "odom"：使用 /odom.pose.pose（与 GoalProcessor 一致，适合 odom≈world 或你只关心相对起点的边界）
    # - "world"：使用 TF(world->base_link) 平移（更适合 Gazebo 里“离开地图区域/离开训练场地”的判定）
    oob_pose_frame: str = "odom"
    oob_world_frame: str = "world"
    oob_base_frame: str = "base_link"
    # 当 TF 树不存在 world->base_link 时（常见于仅发布 odom->base_link），用 Gazebo 服务读取 world 坐标
    # ROS2 Humble：gazebo_ros_state 提供 GetEntityState；GetModelState 多为遗留/不可用
    get_entity_state_services: tuple[str, ...] = (
        "/get_entity_state",
        "/gazebo/get_entity_state",
    )
    get_model_state_services: tuple[str, ...] = (
        "/get_model_state",
        "/gazebo/get_model_state",
    )

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

    wait_ready_timeout_sec: float = 45.0
    """reset 后等待 /scan、/odom、/goal_pose 就绪的最长时间。PPO update 阶段不 spin 时易积压，10s 常不够。"""

    give_success_reward: bool = True
    """是否启用到达目标的一次性奖励（见 ``RewardConfig.w_goal_success``）。"""

    spawn_yaw_towards_goal: bool = False
    """若为 True，每次 ``reset`` 在采样目标后，将车体 yaw 设为指向目标（阶段2/3 避障推荐）。"""

    use_avoid_course_goals: bool = False
    """若为 True，终点由 ``avoid_course.goal_sampling`` 采样（墙内可通行、训练区不含测试区）。"""
    avoid_course_curriculum: bool = False
    """若为 True，按 episode 难度曲线采样终点（需配合 ``use_avoid_course_goals``）。"""
    curriculum_total_episodes: int = 50_000
    """课程难度分母（仅 avoid_course_curriculum 时使用）。"""

    # ---- 日志：每一步写入 CSV（包含 action + 观测关键信息 + reward 分项）----
    step_log_csv: Optional[str] = None
    """例如 'logs/ppo_steps.csv'。None 表示不记录。"""

    linear_cmd_scale_with_front_lidar: bool = True
    """True：cmd.linear.x *= sigmoid(k*(d_front-mid))，用上一步雷达前向最小距 d_front。"""
    linear_cmd_sigmoid_k: float = 8.0
    linear_cmd_sigmoid_mid_m: float = 0.45

    # ---- Level1：随机静态障碍（可选）----
    enable_level1_static_obstacles: bool = False
    static_obstacle_count_min: int = 3
    static_obstacle_count_max: int = 5
    static_obstacle_min_dist_to_robot: float = 1.5
    static_obstacle_min_dist_to_goal: float = 1.5
    static_obstacle_min_dist_between: float = 1.0
    static_obstacle_name_prefix: str = "train_obstacle"
    # Gazebo spawn/delete 服务名候选（classic）
    spawn_entity_services: tuple[str, ...] = ("/spawn_entity", "/gazebo/spawn_entity")
    delete_entity_services: tuple[str, ...] = ("/delete_entity", "/gazebo/delete_entity")


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

        # Gazebo / gazebo_ros 激光与里程计通常为 BEST_EFFORT（SensorDataQoS）。
        # 若订阅用 RELIABLE，与发布端不匹配时整局训练都收不到消息，表现为 scan/odom 永远 None。
        self.create_subscription(LaserScan, cfg.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, cfg.odom_topic, self._on_odom, qos_profile_sensor_data)
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

    def missing_obs_components(self) -> list[str]:
        """用于超时诊断：尚未收到哪些话题/数据。"""
        out: list[str] = []
        if self._scan is None:
            out.append("scan")
        if self._odom is None:
            out.append("odom")
        if self._goal is None:
            out.append("goal")
        if self._obs_cfg.include_camera and self._camera_np is None:
            out.append("camera")
        return out

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

    def get_lidar_angle_params(self) -> tuple[float, float] | None:
        """返回 (angle_min, angle_increment)；无 scan 时 None。"""
        if self._scan is None:
            return None
        return float(self._scan.angle_min), float(self._scan.angle_increment)

    def get_lidar_range_max(self) -> float | None:
        """LaserScan.range_max，用于区分满量程与真实近距离回波；无 scan 时 None。"""
        if self._scan is None:
            return None
        return float(self._scan.range_max)


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
        self._prev_pos_xy: Optional[tuple[float, float]] = None
        self._last_lidar_front_min: Optional[float] = None
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
                "lidar_front_min",
                "done",
                "terminated",
                "truncated",
                "terminal_reason",
                # reward breakdown (flatten)
                "reward_total",
                "reward_progress",
                "reward_direction",
                "reward_safe",
                "reward_risk",
                "reward_front_safe",
                "reward_front_risk",
                "reward_turn",
                "reward_stop",
                "reward_smooth",
                "reward_terminal",
            ] + [f"lidar_feat_{i}" for i in range(int(self.spec.observation_config.lidar_dim))]
            self._step_log_writer.writerow(header)
            self._step_log_f.flush()

        self._set_model_client: Any = None
        self._set_entity_client: Any = None
        self._get_model_state_client: Any = None
        self._get_entity_state_client: Any = None
        self._reset_world_client: Any = None
        self._gazebo_msgs_ok = False
        self._set_model_resolve_warned = False
        self._disable_set_model_resolution = False
        self._disable_set_entity_resolution = False
        self._disable_get_model_state_resolution = False
        self._disable_get_entity_state_resolution = False
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

        # ---- TF（可选）：用于 world 坐标系下的越界判定 ----
        self._tf_buffer = None
        self._tf_listener = None
        self._world_xy_fallback_warned = False
        if str(self.env_cfg.oob_pose_frame).lower() == "world":
            try:
                import tf2_ros  # type: ignore[import-untyped]

                self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
            except Exception as e:
                self._tf_buffer = None
                self._tf_listener = None
                try:
                    self._node.get_logger().warn(
                        f"无法初始化 tf2_ros（将回退为 odom 越界判定）：{e}"
                    )
                except Exception:
                    pass

        # ---- 障碍物场景管理器（可选）----
        self._obstacle_mgr: GazeboObstacleManager | None = None
        if bool(self.env_cfg.enable_level1_static_obstacles):
            self._obstacle_mgr = GazeboObstacleManager(
                node=self._node,
                cfg=GazeboObstacleManagerConfig(
                    spawn_entity_services=tuple(self.env_cfg.spawn_entity_services),
                    delete_entity_services=tuple(self.env_cfg.delete_entity_services),
                ),
            )

    def _pump(self, n: int = 15, timeout_sec: float = 0.02) -> None:
        for _ in range(n):
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def _wait_ready(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = float(self.env_cfg.wait_ready_timeout_sec)
        # 先排空队列：上一轮 PPO update 可能数十秒未 spin，此处易误判超时
        self._pump(120, 0.02)
        t0 = time.time()
        while time.time() - t0 < float(timeout):
            self._pump(8, 0.02)
            if self._node.ready_for_obs():
                return
            time.sleep(0.02)
        missing = ", ".join(self._node.missing_obs_components()) or "unknown"
        raise TimeoutError(
            f"等待 /scan /odom /goal_pose 超时（{timeout:g}s），缺失: [{missing}]。"
            " 若发生在多轮训练中途：多为 update() 期间未处理 ROS 回调；已加大默认超时并在 collect_rollout 前增加 spin。"
            " 请确认 gzserver 仍在运行。"
        )

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

    def _call_set_model_state(self, yaw: Optional[float] = None) -> bool:
        if not self._resolve_set_model_client():
            return False
        try:
            from gazebo_msgs.msg import ModelState  # type: ignore[import-untyped]
            from gazebo_msgs.srv import SetModelState  # type: ignore[import-untyped]
        except ImportError:
            return False

        yaw_use = float(self.env_cfg.spawn_yaw) if yaw is None else float(yaw)
        ms = ModelState()
        ms.model_name = self.env_cfg.model_name
        ms.pose.position.x = float(self.env_cfg.spawn_x)
        ms.pose.position.y = float(self.env_cfg.spawn_y)
        ms.pose.position.z = float(self.env_cfg.spawn_z)
        ms.pose.orientation = _yaw_to_quat(yaw_use)
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

    def _call_set_entity_state(self, yaw: Optional[float] = None) -> bool:
        if not self._resolve_set_entity_client():
            return False
        try:
            from gazebo_msgs.msg import EntityState  # type: ignore[import-untyped]
            from gazebo_msgs.srv import SetEntityState  # type: ignore[import-untyped]
        except ImportError:
            return False

        yaw_use = float(self.env_cfg.spawn_yaw) if yaw is None else float(yaw)
        st = EntityState()
        st.name = self.env_cfg.model_name
        st.pose.position.x = float(self.env_cfg.spawn_x)
        st.pose.position.y = float(self.env_cfg.spawn_y)
        st.pose.position.z = float(self.env_cfg.spawn_z)
        st.pose.orientation = _yaw_to_quat(yaw_use)
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

    def _discover_get_entity_state_service_names(self) -> list[str]:
        target = "gazebo_msgs/srv/GetEntityState"
        found: list[str] = []
        try:
            for name, types in self._node.get_service_names_and_types():
                if target in types:
                    found.append(name)
        except Exception:
            pass
        return sorted(found)

    def _resolve_get_entity_state_client(self) -> bool:
        if self._disable_get_entity_state_resolution:
            return False
        if self._get_entity_state_client is not None:
            return True
        if not self._gazebo_msgs_ok:
            return False
        try:
            from gazebo_msgs.srv import GetEntityState  # type: ignore[import-untyped]
        except ImportError:
            return False

        self._pump(20, 0.02)
        discovered = self._discover_get_entity_state_service_names()
        seen: set[str] = set()
        candidates: list[str] = []
        for s in discovered + list(self.env_cfg.get_entity_state_services):
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(GetEntityState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._get_entity_state_client = cli
                    self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（GetEntityState）")
                    return True
                self._pump(15, 0.02)
            try:
                self._node.destroy_client(cli)
            except Exception:
                pass

        self._disable_get_entity_state_resolution = True
        return False

    def _discover_get_model_state_service_names(self) -> list[str]:
        target = "gazebo_msgs/srv/GetModelState"
        found: list[str] = []
        try:
            for name, types in self._node.get_service_names_and_types():
                if target in types:
                    found.append(name)
        except Exception:
            pass
        return sorted(found)

    def _resolve_get_model_state_client(self) -> bool:
        if self._disable_get_model_state_resolution:
            return False
        if self._get_model_state_client is not None:
            return True
        if not self._gazebo_msgs_ok:
            return False
        try:
            from gazebo_msgs.srv import GetModelState  # type: ignore[import-untyped]
        except ImportError:
            return False

        self._pump(20, 0.02)
        discovered = self._discover_get_model_state_service_names()
        seen: set[str] = set()
        candidates: list[str] = []
        for s in discovered + list(self.env_cfg.get_model_state_services):
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(GetModelState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._get_model_state_client = cli
                    self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（GetModelState）")
                    return True
                self._pump(15, 0.02)
            try:
                self._node.destroy_client(cli)
            except Exception:
                pass

        self._disable_get_model_state_resolution = True
        return False

    def _get_world_xy_via_get_entity_state(self) -> tuple[float, float] | None:
        """ROS2：``gazebo_ros_state`` 插件提供的 ``GetEntityState``（优先于已弃用的 GetModelState）。"""
        if not self._resolve_get_entity_state_client():
            return None
        try:
            from gazebo_msgs.srv import GetEntityState  # type: ignore[import-untyped]
        except ImportError:
            return None

        self._pump(20, 0.02)
        req = GetEntityState.Request()
        req.name = str(self.env_cfg.model_name)
        req.reference_frame = "world"
        fut = self._get_entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None:
            return None
        if hasattr(res, "success") and (not res.success):
            return None
        try:
            p = res.state.pose.position
            return float(p.x), float(p.y)
        except Exception:
            return None

    def _get_world_xy_via_get_model_state_legacy(self) -> tuple[float, float] | None:
        """遗留 ``GetModelState``（多数 ROS2 安装中已不提供对应 service）。"""
        if not self._resolve_get_model_state_client():
            return None
        try:
            from gazebo_msgs.srv import GetModelState  # type: ignore[import-untyped]
        except ImportError:
            return None

        self._pump(20, 0.02)
        req = GetModelState.Request()
        req.model_name = str(self.env_cfg.model_name)
        req.relative_entity_name = "world"
        fut = self._get_model_state_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None:
            return None
        if hasattr(res, "success") and (not res.success):
            return None
        try:
            p = res.pose.position
            return float(p.x), float(p.y)
        except Exception:
            return None

    def _get_world_xy_via_gazebo_model_state(self) -> tuple[float, float] | None:
        """
        在 TF 无 ``world->base_link`` 时，用 Gazebo 查询 ``model_name`` 的 world 平面坐标 ``(x,y)``。

        ROS2 Humble 下应优先走 ``GetEntityState``（需在 world 中加载 ``libgazebo_ros_state.so``）；
        ``GetModelState`` 仅作兼容回退。
        """
        xy = self._get_world_xy_via_get_entity_state()
        if xy is not None:
            return xy
        return self._get_world_xy_via_get_model_state_legacy()

    def _sample_goal_xy(self) -> tuple[float, float]:
        if self.env_cfg.use_avoid_course_goals:
            from obstacle_environment.avoid_course.goal_sampling import (
                GoalSamplingConfig,
                sample_goal_curriculum,
                sample_random_goal,
            )

            cfg = GoalSamplingConfig(
                allow_test_region=False,
                map_x_min=float(self.env_cfg.map_x_min),
                map_x_max=float(self.env_cfg.map_x_max),
                map_y_min=float(self.env_cfg.map_y_min),
                map_y_max=float(self.env_cfg.map_y_max),
                min_distance_from_start=float(self.env_cfg.goal_min_distance),
            )
            start = (float(self.env_cfg.spawn_x), float(self.env_cfg.spawn_y))
            g = None
            if self.env_cfg.avoid_course_curriculum:
                g = sample_goal_curriculum(
                    int(self._episode_idx),
                    int(self.env_cfg.curriculum_total_episodes),
                    start,
                    cfg,
                )
            else:
                g = sample_random_goal(start, cfg, max_tries=600)
            if g is not None:
                return float(g[0]), float(g[1])
            self._node.get_logger().warn(
                "avoid_course 终点采样失败，回退为矩形内随机（可能穿墙，建议检查地图与边界）"
            )

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
    def _goal_xy_from_obs(obs: dict[str, Any]) -> tuple[float, float]:
        g = obs.get("goal")
        if isinstance(g, dict):
            arr = g.get("goal_xy")
            if isinstance(arr, np.ndarray) and arr.size >= 2:
                return float(arr[0]), float(arr[1])
        return (0.0, 0.0)

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

    def _extract_world_xy_base_link(self) -> tuple[float, float] | None:
        """
        获取 base_link 原点在 **Gazebo world** 平面下的 ``(x,y)``，用于越界判定。

        顺序：**先 Gazebo** ``GetEntityState`` / ``GetModelState``（与仅有 ``odom->base_link`` 的常见 TF 一致），
        再尝试 TF ``{oob_world_frame}->{oob_base_frame}``。避免先查 TF 失败产生误导性 WARN。
        """
        gz = self._get_world_xy_via_gazebo_model_state()
        if gz is not None:
            return gz

        if self._tf_buffer is None:
            if not self._world_xy_fallback_warned:
                self._world_xy_fallback_warned = True
                try:
                    self._node.get_logger().warn(
                        "Gazebo 位姿查询失败且无 TF buffer：越界将回退 /odom.pose。"
                        " 请确认 world 中已加载 libgazebo_ros_state.so。"
                    )
                except Exception:
                    pass
            return None

        try:
            import tf2_ros  # type: ignore[import-untyped]
            from geometry_msgs.msg import PointStamped
            from tf2_geometry_msgs import do_transform_point  # type: ignore[import-untyped]

            world = str(self.env_cfg.oob_world_frame)
            base = str(self.env_cfg.oob_base_frame)

            self._pump(30, 0.02)

            pt = PointStamped()
            pt.header.frame_id = base
            pt.header.stamp = rclpy.time.Time().to_msg()
            pt.point.x = 0.0
            pt.point.y = 0.0
            pt.point.z = 0.0

            tf = self._tf_buffer.lookup_transform(world, base, rclpy.time.Time())
            out = do_transform_point(pt, tf)
            return float(out.point.x), float(out.point.y)
        except Exception:
            if not self._world_xy_fallback_warned:
                self._world_xy_fallback_warned = True
                try:
                    self._node.get_logger().warn(
                        "Gazebo GetEntityState/GetModelState 与 TF world->base_link 均失败，"
                        "越界判定将回退 /odom.pose。请确认 world 含 libgazebo_ros_state.so，"
                        "或发布 world 相关 TF。"
                    )
                except Exception:
                    pass
            return None

    def _position_xy_for_oob(self, obs: dict[str, Any]) -> tuple[float, float] | None:
        mode = str(self.env_cfg.oob_pose_frame).lower()
        if mode == "world":
            w = self._extract_world_xy_base_link()
            if w is not None:
                return w
        return self._extract_position_xy_from_obs(obs)

    def _compute_front_lidar_min(self, lidar_ranges: np.ndarray) -> float:
        rcfg = self.spec.reward_config
        lidar_rm = self._node.get_lidar_range_max()
        dmin = lidar_min_range(lidar_ranges, range_max=lidar_rm)
        ap = self._node.get_lidar_angle_params()
        if ap is not None:
            amin, ainc = ap
            return lidar_front_min_range(
                lidar_ranges,
                angle_min=amin,
                angle_increment=ainc,
                half_width_rad=float(rcfg.front_sector_half_width_rad),
                range_max=lidar_rm,
            )
        return dmin

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

    def spin_ros(self, n: int = 120, timeout_sec: float = 0.02) -> None:
        """主动处理订阅回调（在长时间非 ROS 计算后调用，例如每次 collect_rollout 开始前）。"""
        self._pump(int(n), float(timeout_sec))

    def reset(self) -> np.ndarray:
        """停车、重置位姿与目标、清空轨迹变量，返回初始 ``state`` 向量。"""
        self._pump(60, 0.02)
        self._node.stop_robot()
        gx, gy = self._sample_goal_xy()
        yaw_reset: Optional[float] = None
        if self.env_cfg.spawn_yaw_towards_goal:
            yaw_reset = float(
                math.atan2(gy - float(self.env_cfg.spawn_y), gx - float(self.env_cfg.spawn_x))
            )
        # 优先 set_model_state；失败时退化到 /reset_world，避免一直从错误姿态继续跑飞
        ok = self._call_set_model_state(yaw_reset)
        if not ok:
            ok = self._call_set_entity_state(yaw_reset)
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

        self._node.publish_goal(gx, gy)

        # ---- Level1：随机静态障碍（在每个 episode reset 时刷新）----
        if self._obstacle_mgr is not None and bool(self.env_cfg.enable_level1_static_obstacles):
            try:
                # 先清理上一轮生成的障碍物（如果 DeleteEntity 可用）
                self._obstacle_mgr.clear_spawned()

                n_min = int(self.env_cfg.static_obstacle_count_min)
                n_max = int(self.env_cfg.static_obstacle_count_max)
                if n_max < n_min:
                    n_max = n_min
                n_obs = int(np.random.randint(n_min, n_max + 1))

                samp_cfg = StaticObstacleSamplingConfig(
                    x_min=float(self.env_cfg.map_x_min),
                    x_max=float(self.env_cfg.map_x_max),
                    y_min=float(self.env_cfg.map_y_min),
                    y_max=float(self.env_cfg.map_y_max),
                    min_dist_to_robot=float(self.env_cfg.static_obstacle_min_dist_to_robot),
                    min_dist_to_goal=float(self.env_cfg.static_obstacle_min_dist_to_goal),
                    min_dist_between_obstacles=float(self.env_cfg.static_obstacle_min_dist_between),
                )
                start_xy = (float(self.env_cfg.spawn_x), float(self.env_cfg.spawn_y))
                goal_xy = (float(gx), float(gy))
                specs = sample_static_obstacles(
                    num_obstacles=n_obs,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    cfg=samp_cfg,
                    name_prefix=str(self.env_cfg.static_obstacle_name_prefix),
                )
                obstacles = [(s.name, float(s.x), float(s.y), float(s.yaw)) for s in specs]
                self._obstacle_mgr.spawn_static_boxes(obstacles=obstacles)
            except Exception as e:
                try:
                    self._node.get_logger().warn(f"静态障碍刷新失败（将继续训练，不影响 reset）：{e}")
                except Exception:
                    pass

        self._prev_goal_distance = None
        self._prev_pos_xy = None
        self._last_lidar_front_min = None
        self._step_idx = 0
        self._goal_reached_for_bonus = False
        self._prev_action = None
        self._episode_idx += 1

        time.sleep(self.env_cfg.reset_settle_time)
        self._wait_ready()

        obs = self._node.build_observation()
        self._prev_goal_distance = float(obs["goal_distance"])
        self._prev_pos_xy = self._extract_position_xy_from_obs(obs)
        lr0 = self._node.get_lidar_ranges()
        if lr0.size > 0:
            self._last_lidar_front_min = float(self._compute_front_lidar_min(lr0))
        return obs["state"].astype(np.float32, copy=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Returns:
            state, reward, done, info
        """
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        a0 = float(raw_action[0]) if raw_action.size > 0 else 0.0
        a1 = float(raw_action[1]) if raw_action.size > 1 else 0.0

        cmd = np.asarray(self._mapper.to_linear_angular(raw_action), dtype=np.float32).reshape(-1)
        # 注意：若激光读数不可靠（常见：全 inf / 无回波），不要用上一步的 d_front 去缩放线速度，
        # 否则可能把线速度压到接近 0，表现为“原地打转”。
        if (
            self.env_cfg.linear_cmd_scale_with_front_lidar
            and self._last_lidar_front_min is not None
            and float(self._last_lidar_front_min) > 1e-3
        ):
            sc = _linear_cmd_sigmoid_scale(
                float(self._last_lidar_front_min),
                k=float(self.env_cfg.linear_cmd_sigmoid_k),
                mid_m=float(self.env_cfg.linear_cmd_sigmoid_mid_m),
            )
            cmd[0] = float(cmd[0]) * sc
        self._node.publish_cmd_vel(float(cmd[0]), float(cmd[1]))

        time.sleep(self.env_cfg.control_dt)
        self._pump(20, 0.02)

        obs = self._node.build_observation()
        state = obs["state"].astype(np.float32, copy=False)
        lidar_ranges = self._node.get_lidar_ranges()
        gd = float(obs["goal_distance"])
        rcfg = self.spec.reward_config
        lidar_rm = self._node.get_lidar_range_max()

        dmin = lidar_min_range(lidar_ranges, range_max=lidar_rm)
        lidar_ok_for_collision = _lidar_has_reliable_close_reading(
            lidar_ranges,
            range_max=lidar_rm,
            collision_distance=float(rcfg.collision_distance),
        )
        ap = self._node.get_lidar_angle_params()
        if ap is not None:
            amin, ainc = ap
            d_front = lidar_front_min_range(
                lidar_ranges,
                angle_min=amin,
                angle_increment=ainc,
                half_width_rad=float(rcfg.front_sector_half_width_rad),
                range_max=lidar_rm,
            )
        else:
            d_front = dmin
        self._last_lidar_front_min = float(d_front)
        pos_xy = self._extract_position_xy_from_obs(obs)
        oob_xy = self._position_xy_for_oob(obs)
        out_of_bounds = self._is_out_of_bounds(oob_xy)
        terminated = False
        terminal_reason: Optional[str] = None
        if (
            self.env_cfg.done_on_collision
            and lidar_ok_for_collision
            and dmin <= rcfg.collision_distance
        ):
            terminated = True
            terminal_reason = "collision"
        if (not terminated) and self.env_cfg.done_on_goal and gd <= rcfg.goal_reached_distance:
            terminated = True
            terminal_reason = "success"
        if (not terminated) and self.env_cfg.done_on_out_of_bounds and out_of_bounds:
            terminated = True
            terminal_reason = "out_of_bounds"

        gx, gy = self._goal_xy_from_obs(obs)
        rb = compute_reward(
            lidar_ranges=lidar_ranges,
            linear_x=float(obs["linear_x"]),
            angular_z=float(obs.get("angular_z", 0.0)),
            goal_distance=gd,
            goal_angle=float(obs.get("goal_angle", 0.0)),
            lidar_front_min=float(d_front),
            lidar_range_max=lidar_rm,
            prev_goal_distance=self._prev_goal_distance,
            prev_robot_xy=self._prev_pos_xy,
            robot_xy=pos_xy if pos_xy is not None else None,
            goal_x=gx,
            goal_y=gy,
            action=cmd,
            prev_action=self._prev_action,
            config=rcfg,
            terminal=terminal_reason,
        )

        self._prev_action = cmd.astype(np.float32, copy=False)
        self._prev_goal_distance = gd
        if pos_xy is not None:
            self._prev_pos_xy = pos_xy
        self._step_idx += 1

        truncated = self._step_idx >= self.env_cfg.max_episode_steps
        done = terminated or truncated

        info = {
            "reward_breakdown": rb,
            "terminated": terminated,
            "truncated": truncated,
            "lidar_min_m": dmin,
            "lidar_front_min_m": float(d_front),
            "lidar_ok_for_collision": bool(lidar_ok_for_collision),
            "goal_distance": gd,
            "position_xy": pos_xy,
            "oob_position_xy": oob_xy,
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
                f"{float(d_front):.6f}",
                "1" if done else "0",
                "1" if terminated else "0",
                "1" if truncated else "0",
                terminal_reason or "",
                f"{rb_dict.get('reward_total', 0.0):.6f}",
                f"{rb_dict.get('reward_progress', 0.0):.6f}",
                f"{rb_dict.get('reward_direction', 0.0):.6f}",
                f"{rb_dict.get('reward_safe', 0.0):.6f}",
                f"{rb_dict.get('reward_risk', 0.0):.6f}",
                f"{rb_dict.get('reward_front_safe', 0.0):.6f}",
                f"{rb_dict.get('reward_front_risk', 0.0):.6f}",
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
