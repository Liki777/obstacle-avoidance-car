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
from obstacle_environment.reward.reward_computer import _finite_non_saturating
from obstacle_environment.robot_spec import RobotTaskSpec
from obstacle_environment.scenario_manager import GazeboObstacleManager, GazeboObstacleManagerConfig
from obstacle_environment.world_generator import StaticObstacleSamplingConfig, sample_static_obstacles
from obstacle_environment.world_generator.dynamic_obstacle_presets import (
    DynamicObstacleSpec,
    builtin_fixed_dynamic_specs,
    builtin_level1_fixed_mixed_3x3,
    pose_at_time,
    sample_random_dynamic_specs,
)


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
    # 终止用「前向扇区」裕量：仅用于车头 ±front_sector 内最近距 d_front（见 step 内 lidar_front_min）。
    # 真实蹭墙时往往是前向读数偏小；若把同一裕量用到全向 min(dmin)，侧身掠过障碍时侧向射线仍 ~0.30m
    # 会误触发 collision（肉眼已过、日志 dmin≈0.30）。
    collision_done_extra_margin_m: float = 0.14
    # 全向最近距（非饱和射线 min）在 collision_distance 上再加一小段：覆盖侧向刮擦，又避免 0.30m 级「擦边通过」误判
    collision_done_global_small_margin_m: float = 0.055
    # 激光未可靠命中墙体（例如全向开阔但物理已卡住）时的兜底：持续前进指令 + 里程计几乎不动 + 周向最近距足够近
    done_on_cmd_velocity_stuck: bool = True
    stuck_cmd_linear_min: float = 0.06
    stuck_odom_linear_max: float = 0.028
    # 卡死判据：全向最近距 ≤ 该值（米）时认为近障，配合「有前进指令但里程计不动」
    stuck_lidar_max_m: float = 0.72
    stuck_consecutive_steps: int = 8
    # 另一种卡死兜底：位置几乎不动（更适合“贴路沿石/墙体挤住但雷达不一定足够近”的情况）
    stuck_pos_eps_m: float = 0.004
    """若连续若干步位移 < eps 且有前进指令，则判定卡死。"""
    stuck_pos_consecutive_steps: int = 6
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

    debug_reset: bool = False
    """True：每次 reset 打印重置路径与位姿偏差（用于验证 reset 是否真的生效）。"""

    # ---- Road-following task (Level0 road) ----
    road_map_yaml: str = ""
    """道路中心线配置（YAML 路径）。留空表示不启用道路特征/道路奖励。"""
    road_half_width_m: float = 0.65
    """道路半宽（用于 out-of-road 判定）。若 YAML 内提供 half_width_m 会覆盖此值。"""
    road_lookahead_n: int = 5
    road_lookahead_ds: float = 0.8
    road_out_margin_m: float = 0.05
    done_on_out_of_road: bool = True
    out_of_road_penalty: float = 60.0
    done_on_road_end: bool = True
    road_success_s_margin_m: float = 0.8
    # 与 Gazebo world 里平面 goal marker（绿圈）几何对齐：用车体 world 位姿到 (goal_x,goal_y) 判 success
    done_on_world_goal: bool = False
    world_goal_success_radius_m: float = 0.40
    """world 平面 dist(车体, goal) ≤ 该值则 terminal_reason=success（常见 marker 半径 0.30，略留裕量）。"""

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

    # ---- 静态障碍（课程 Level0/1/2；与 map_design_document 一致）----
    # ``none`` | ``fixed`` | ``random``；若仍设置 ``enable_level1_static_obstacles=True`` 且本字段为 ``none``，环境内会退化为 ``random``（兼容旧入口）。
    static_obstacle_mode: str = "none"
    fixed_static_obstacles_xyyaw: tuple[tuple[float, float, float], ...] = ()
    static_obstacle_count_min: int = 3
    static_obstacle_count_max: int = 5
    static_obstacle_min_dist_to_robot: float = 1.5
    static_obstacle_min_dist_to_goal: float = 1.5
    static_obstacle_min_dist_between: float = 1.0
    static_obstacle_name_prefix: str = "train_obstacle"
    enable_level1_static_obstacles: bool = False
    """兼容旧版：等价于 ``static_obstacle_mode="random"``（当 mode 仍为 none 时）。"""

    # ---- 动态障碍（Level3/4/5）：每步 SetEntityState 更新位姿 ----
    dynamic_obstacle_mode: str = "none"
    """``none`` | ``fixed`` | ``random``。"""
    dynamic_obstacle_count_min: int = 1
    dynamic_obstacle_count_max: int = 2
    dynamic_obstacle_name_prefix: str = "train_dyn"

    # Gazebo spawn/delete/set 服务名候选（classic）
    spawn_entity_services: tuple[str, ...] = ("/spawn_entity", "/gazebo/spawn_entity")
    delete_entity_services: tuple[str, ...] = ("/delete_entity", "/gazebo/delete_entity")
    set_entity_state_services: tuple[str, ...] = ("/set_entity_state", "/gazebo/set_entity_state")


class _RlCarGymNode(Node):
    """订阅传感器 + 发布 ``cmd_vel`` 与 ``goal_pose``。"""

    def __init__(
        self,
        *,
        cfg: GazeboEnvConfig,
        obs_cfg: ObservationConfig,
    ) -> None:
        # 并行训练/多进程或重复创建环境时，固定节点名会触发 rcl.logging_rosout
        # “Publisher already registered for provided node name” 警告。使用 pid 生成唯一名可避免。
        super().__init__(f"rl_car_gym_env_{os.getpid()}")
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

    def build_observation(self, *, road_data: Any = None) -> dict[str, Any]:
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
            road_data=road_data,
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
        self._collision_stuck_steps = 0
        self._stuck_pos_steps = 0
        self._road_map = None
        self._prev_road_s: Optional[float] = None
        self._road_s_end: Optional[float] = None

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
        self._set_entity_service_name: str = ""
        self._set_model_service_name: str = ""
        self._get_model_state_client: Any = None
        self._get_entity_state_client: Any = None
        self._reset_world_client: Any = None
        self._gazebo_msgs_ok = False
        self._set_model_resolve_warned = False
        self._set_entity_resolve_warned = False
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

        # ---- 障碍物场景管理器（静态 / 动态 spawn 与动态位姿更新）----
        self._obstacle_mgr: GazeboObstacleManager | None = None
        self._dynamic_specs: list[DynamicObstacleSpec] = []
        self._dynamic_episode_t: float = 0.0

        st_mode = str(self.env_cfg.static_obstacle_mode).strip().lower()
        if bool(self.env_cfg.enable_level1_static_obstacles) and st_mode in ("", "none"):
            st_mode = "random"
        dyn_mode = str(self.env_cfg.dynamic_obstacle_mode).strip().lower()
        need_obstacle_mgr = (st_mode in ("fixed", "random")) or (dyn_mode == "fixed") or (
            dyn_mode == "random" and int(self.env_cfg.dynamic_obstacle_count_max) > 0
        )
        if need_obstacle_mgr:
            self._obstacle_mgr = GazeboObstacleManager(
                node=self._node,
                cfg=GazeboObstacleManagerConfig(
                    spawn_entity_services=tuple(self.env_cfg.spawn_entity_services),
                    delete_entity_services=tuple(self.env_cfg.delete_entity_services),
                    set_entity_state_services=tuple(self.env_cfg.set_entity_state_services),
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

        if not candidates:
            # 启动早期可能尚未注册完成；后续 reset 仍会重试
            return False

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(SetModelState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._set_model_client = cli
                    self._set_model_service_name = str(svc)
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
                " 将改用 SetEntityState（若可用）；若两者都不可用，才会跳过位姿重置。"
            )
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

        if not candidates:
            # 启动早期可能尚未注册完成，允许后续重试
            return False

        wait_each = float(self.env_cfg.set_model_state_wait_sec)
        for svc in candidates:
            cli = self._node.create_client(SetEntityState, svc)
            t0 = time.time()
            while time.time() - t0 < wait_each:
                if cli.service_is_ready():
                    self._set_entity_client = cli
                    self._set_entity_service_name = str(svc)
                    self._node.get_logger().info(f"已连接 Gazebo 服务: {svc}（SetEntityState）")
                    return True
                self._pump(15, 0.02)
            try:
                self._node.destroy_client(cli)
            except Exception:
                pass
        if not getattr(self, "_set_entity_resolve_warned", False):
            self._set_entity_resolve_warned = True
            self._node.get_logger().warn(
                "未发现类型为 gazebo_msgs/srv/SetEntityState 的服务（已尝试自动发现 + "
                f"{list(self.env_cfg.set_entity_state_services)}）。将无法通过服务重置位姿（将尝试 /reset_world 或仅停车继续）。"
            )
        # 不永久禁用：服务可能在 Gazebo 完全启动后才出现
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

        # 若实体还未 spawn（常见于 gzserver 启动后延迟 spawn），直接跳过，避免每次 reset 都刷失败警告
        # 这里用 GetEntityState 做一次快速探测（2s 超时）。
        try:
            if self._get_world_xy_via_get_entity_state() is None:
                return False
        except Exception:
            pass

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
            msg = str(getattr(res, "status_message", "") or "").strip()
            svc = str(getattr(self, "_set_entity_service_name", "") or "").strip()
            extra = f" service={svc}" if svc else ""
            # 有些版本 status_message 为空；至少把 entity 名与目标 pose 打出来，便于定位“实体不存在/名字不对/服务不对”
            self._node.get_logger().warn(
                "SetEntityState 失败"
                + (f": {msg}" if msg else "")
                + f"{extra} name={st.name!r} target=({st.pose.position.x:.2f},{st.pose.position.y:.2f},{yaw_use:.2f})"
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

    def _get_world_xy_yaw_via_get_entity_state(self) -> tuple[float, float, float] | None:
        """GetEntityState 同时拿 world (x,y,yaw)，用于与 world 内静态 marker 对齐诊断。"""
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
            q = res.state.pose.orientation
            # yaw from quaternion (Z axis)
            siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
            cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
            yaw = float(np.arctan2(siny_cosp, cosy_cosp))
            return float(p.x), float(p.y), yaw
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
        # road map lazy-load (optional)
        if self._road_map is None and str(self.env_cfg.road_map_yaml).strip():
            from obstacle_environment.road import RoadMap
            self._road_map = RoadMap.load(str(self.env_cfg.road_map_yaml))
            # allow YAML to override half width
            try:
                self.env_cfg.road_half_width_m = float(self._road_map.half_width_m)
            except Exception:
                pass
            try:
                self._road_s_end = float(self._road_map.s_end)
            except Exception:
                self._road_s_end = None
        gx, gy = self._sample_goal_xy()
        yaw_reset: Optional[float] = None
        if self.env_cfg.spawn_yaw_towards_goal:
            yaw_reset = float(
                math.atan2(gy - float(self.env_cfg.spawn_y), gx - float(self.env_cfg.spawn_x))
            )
        # 优先 set_model_state；失败时退化到 set_entity_state / reset_world，避免一直从错误姿态继续跑飞
        ok = False
        reset_via = "none"
        if self._call_set_model_state(yaw_reset):
            ok = True
            reset_via = "set_model_state"
        elif self._call_set_entity_state(yaw_reset):
            ok = True
            reset_via = "set_entity_state"

        if (not ok) and (self._reset_world_client is not None):
            try:
                from std_srvs.srv import Empty  # type: ignore[import-untyped]

                # 启动早期/并行时服务发现可能滞后：边 pump 边等
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    if self._reset_world_client.wait_for_service(timeout_sec=0.2):
                        break
                    self._pump(10, 0.02)
                if self._reset_world_client.service_is_ready():
                    fut = self._reset_world_client.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(self._node, fut, timeout_sec=3.0)
                    ok = True
                    reset_via = "reset_world"
            except Exception:
                pass
        if (not ok) and (not self._set_model_resolve_warned):
            self._set_model_resolve_warned = True
            self._node.get_logger().warn(
                "未发现 set_model_state/set_entity_state，且 /reset_world 不可用；重置将仅发布零速度并继续。"
            )

        if bool(self.env_cfg.debug_reset):
            try:
                # 先 pump 一下，让 /odom 追上服务重置后的位姿
                self._pump(40, 0.02)
                obs0 = self._node.build_observation()
                pos0 = self._extract_position_xy_from_obs(obs0)
                if pos0 is not None:
                    dx = float(pos0[0]) - float(self.env_cfg.spawn_x)
                    dy = float(pos0[1]) - float(self.env_cfg.spawn_y)
                    dist = float(np.hypot(dx, dy))
                    self._node.get_logger().info(
                        f"[debug_reset] via={reset_via} ok={ok} "
                        f"odom_xy=({pos0[0]:.3f},{pos0[1]:.3f}) "
                        f"spawn_xy=({float(self.env_cfg.spawn_x):.3f},{float(self.env_cfg.spawn_y):.3f}) "
                        f"err={dist:.3f}m"
                    )
                else:
                    self._node.get_logger().info(f"[debug_reset] via={reset_via} ok={ok} odom_xy=(none)")
            except Exception as e:
                try:
                    self._node.get_logger().warn(f"[debug_reset] failed: {e}")
                except Exception:
                    pass

        self._node.publish_goal(gx, gy)

        # ---- 静态 + 动态障碍（每个 episode reset）----
        self._dynamic_specs = []
        self._dynamic_episode_t = 0.0
        if self._obstacle_mgr is not None:
            try:
                self._obstacle_mgr.clear_spawned()

                st_mode = str(self.env_cfg.static_obstacle_mode).strip().lower()
                if bool(self.env_cfg.enable_level1_static_obstacles) and st_mode in ("", "none"):
                    st_mode = "random"

                if st_mode == "random":
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
                elif st_mode == "fixed":
                    px = str(self.env_cfg.static_obstacle_name_prefix)
                    fix = list(self.env_cfg.fixed_static_obstacles_xyyaw)
                    if not fix:
                        # 跨进程运行 eval/train 时，上一轮进程生成的固定障碍可能仍残留在 Gazebo 中。
                        # clear_spawned() 只清理当前进程记录的名字，因此这里额外按约定命名做一次“尽力删除”，避免一开局就撞。
                        try:
                            self._obstacle_mgr.delete_entities(
                                names=[f"{px}_l1_{i:02d}" for i in range(9)]
                            )
                        except Exception:
                            pass
                        self._obstacle_mgr.spawn_mixed_static(
                            specs=builtin_level1_fixed_mixed_3x3(name_prefix=f"{px}_l1"),
                        )
                    else:
                        obstacles = [
                            (f"{px}_fixed_{i}", float(x), float(y), float(yaw))
                            for i, (x, y, yaw) in enumerate(fix)
                        ]
                        self._obstacle_mgr.spawn_static_boxes(obstacles=obstacles)

                dyn_mode = str(self.env_cfg.dynamic_obstacle_mode).strip().lower()
                if dyn_mode == "fixed":
                    self._dynamic_specs = list(
                        builtin_fixed_dynamic_specs(
                            name_prefix=str(self.env_cfg.dynamic_obstacle_name_prefix)
                        )
                    )
                elif dyn_mode == "random" and int(self.env_cfg.dynamic_obstacle_count_max) > 0:
                    n_min = int(self.env_cfg.dynamic_obstacle_count_min)
                    n_max = int(self.env_cfg.dynamic_obstacle_count_max)
                    if n_max < n_min:
                        n_max = n_min
                    n_dyn = int(np.random.randint(n_min, n_max + 1))
                    import random as _pyr

                    rng = _pyr.Random(int(self._episode_idx) * 1013 + 31)
                    self._dynamic_specs = sample_random_dynamic_specs(
                        rng,
                        count=n_dyn,
                        map_x_min=float(self.env_cfg.map_x_min),
                        map_x_max=float(self.env_cfg.map_x_max),
                        map_y_min=float(self.env_cfg.map_y_min),
                        map_y_max=float(self.env_cfg.map_y_max),
                        start_xy=(float(self.env_cfg.spawn_x), float(self.env_cfg.spawn_y)),
                        goal_xy=(float(gx), float(gy)),
                        name_prefix=str(self.env_cfg.dynamic_obstacle_name_prefix),
                    )

                if self._dynamic_specs:
                    dyn_boxes = [(s.name, *pose_at_time(s, 0.0)) for s in self._dynamic_specs]
                    self._obstacle_mgr.spawn_static_boxes(obstacles=dyn_boxes)
            except Exception as e:
                try:
                    self._node.get_logger().warn(f"障碍刷新失败（将继续训练，不影响 reset）：{e}")
                except Exception:
                    pass

        self._prev_goal_distance = None
        self._prev_pos_xy = None
        self._last_lidar_front_min = None
        self._step_idx = 0
        self._goal_reached_for_bonus = False
        self._prev_action = None
        self._collision_stuck_steps = 0
        self._stuck_pos_steps = 0
        self._prev_road_s = None
        # keep _road_s_end across episodes
        self._episode_idx += 1

        time.sleep(self.env_cfg.reset_settle_time)
        self._wait_ready()

        base = self._node.build_observation()
        road_data = self._compute_road_data(base)
        obs = self._node.build_observation(road_data=road_data)
        self._prev_goal_distance = float(obs["goal_distance"])
        self._prev_pos_xy = self._extract_position_xy_from_obs(obs)
        if self._road_map is not None and isinstance(obs.get("road"), dict):
            try:
                self._prev_road_s = float(obs["road"].get("s", 0.0))
            except Exception:
                self._prev_road_s = None
        lr0 = self._node.get_lidar_ranges()
        if lr0.size > 0:
            self._last_lidar_front_min = float(self._compute_front_lidar_min(lr0))
        return obs["state"].astype(np.float32, copy=False)

    def _yaw_from_odom(self, odom: Any) -> float:
        try:
            q = odom.pose.pose.orientation
            x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return float(math.atan2(siny_cosp, cosy_cosp))
        except Exception:
            return 0.0

    def _compute_road_data(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._road_map is None:
            return {}
        pos = self._extract_position_xy_from_obs(obs)
        odom = obs.get("odom")
        if pos is None or odom is None:
            return {}
        yaw = self._yaw_from_odom(odom)
        s, cte, tan_yaw = self._road_map.project(xy=pos)
        he = float(self._road_map.heading_error(ego_yaw=yaw, tangent_yaw=tan_yaw))
        la = self._road_map.lookahead_points(
            s=s,
            n=int(self.env_cfg.road_lookahead_n),
            ds=float(self.env_cfg.road_lookahead_ds),
        )
        la_body = self._road_map.road_feat_to_body_frame(ego_xy=pos, ego_yaw=yaw, points_xy=la)
        hw = float(self.env_cfg.road_half_width_m)
        margin = float(self.env_cfg.road_out_margin_m)
        in_road = abs(float(cte)) <= max(0.0, hw - margin)
        return {
            "s": float(s),
            "cte": float(cte),
            "heading_error": float(he),
            "in_road": bool(in_road),
            "lookahead_xy": la_body.astype(np.float32, copy=False),
        }

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
        if self._obstacle_mgr is not None and self._dynamic_specs:
            self._dynamic_episode_t += float(self.env_cfg.control_dt)
            for sp in self._dynamic_specs:
                x, y, yaw = pose_at_time(sp, self._dynamic_episode_t)
                self._obstacle_mgr.set_entity_pose(name=sp.name, x=x, y=y, yaw=yaw)
        self._pump(20, 0.02)

        base = self._node.build_observation()
        road_data = self._compute_road_data(base)
        obs = self._node.build_observation(road_data=road_data)
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
        wpose_step: tuple[float, float, float] | None = None
        world_goal_d: float | None = None
        try:
            wpose_step = self._get_world_xy_yaw_via_get_entity_state()
            if wpose_step is not None:
                wx_s, wy_s, _ = wpose_step
                world_goal_d = float(
                    np.hypot(float(self.env_cfg.goal_x) - float(wx_s), float(self.env_cfg.goal_y) - float(wy_s))
                )
        except Exception:
            pass
        terminated = False
        terminal_reason: Optional[str] = None
        success_via: str = ""
        # 碰撞判定（分层，避免「侧身已过障碍、侧向射线仍 0.30m」却用全向宽阈值误 reset）：
        # - lidar_strict：原逻辑（与奖励 collision_distance 一致）
        # - global_small：全向 min ≤ cd + collision_done_global_small_margin_m（约 0.25m 级，刮擦仍判）
        # - front_relaxed：仅前向 d_front ≤ cd + collision_done_extra_margin_m（约 0.33m，兜底盘/激光装位）
        # - stuck：里程计卡死兜底
        collision_via = ""
        if self.env_cfg.done_on_collision:
            cd = float(rcfg.collision_distance)
            m_front = max(0.0, float(self.env_cfg.collision_done_extra_margin_m))
            m_glob = max(0.0, float(self.env_cfg.collision_done_global_small_margin_m))
            cd_front_done = cd + m_front
            cd_global_done = cd + m_glob
            close_ns = _finite_non_saturating(lidar_ranges, range_max=lidar_rm)
            global_small = close_ns.size > 0 and float(np.min(close_ns)) <= cd_global_done + 1e-4
            front_relaxed = float(d_front) <= cd_front_done + 1e-4
            lidar_strict = lidar_ok_for_collision and dmin <= cd + 1e-4
            contact_lidar = lidar_strict or global_small or front_relaxed

            stuck_phys = False
            if bool(self.env_cfg.done_on_cmd_velocity_stuck):
                cmd_lin = abs(float(cmd[0]))
                odom_lin = abs(float(obs.get("linear_x", 0.0)))
                near_obs = float(dmin) <= float(self.env_cfg.stuck_lidar_max_m)
                if (
                    cmd_lin >= float(self.env_cfg.stuck_cmd_linear_min)
                    and odom_lin <= float(self.env_cfg.stuck_odom_linear_max)
                    and near_obs
                ):
                    self._collision_stuck_steps += 1
                else:
                    self._collision_stuck_steps = 0
                need = max(1, int(self.env_cfg.stuck_consecutive_steps))
                if self._collision_stuck_steps >= need:
                    stuck_phys = True
                    self._collision_stuck_steps = 0

                # 位置几乎不动兜底：不依赖 near_obs（路沿石较低时雷达未必“很近”）
                if (not stuck_phys) and (pos_xy is not None) and (self._prev_pos_xy is not None):
                    disp = float(np.hypot(float(pos_xy[0] - self._prev_pos_xy[0]), float(pos_xy[1] - self._prev_pos_xy[1])))
                    if cmd_lin >= float(self.env_cfg.stuck_cmd_linear_min) and disp <= float(self.env_cfg.stuck_pos_eps_m):
                        self._stuck_pos_steps += 1
                    else:
                        self._stuck_pos_steps = 0
                    need2 = max(1, int(self.env_cfg.stuck_pos_consecutive_steps))
                    if self._stuck_pos_steps >= need2:
                        stuck_phys = True
                        self._stuck_pos_steps = 0

            if contact_lidar or stuck_phys:
                terminated = True
                terminal_reason = "collision"
                collision_via = "lidar" if contact_lidar else ("stuck" if stuck_phys else "")
        if (
            (not terminated)
            and bool(self.env_cfg.done_on_world_goal)
            and world_goal_d is not None
            and world_goal_d <= float(self.env_cfg.world_goal_success_radius_m)
        ):
            terminated = True
            terminal_reason = "success"
            success_via = "world_goal_marker"
        if (not terminated) and self.env_cfg.done_on_goal and gd <= rcfg.goal_reached_distance:
            terminated = True
            terminal_reason = "success"
            success_via = "goal_distance"
        # 道路任务：按中心线进度到达终点（避免终点附近“慢慢停下→truncated”）
        if (
            (not terminated)
            and bool(self.env_cfg.done_on_road_end)
            and isinstance(obs.get("road"), dict)
            and (self._road_s_end is not None)
        ):
            try:
                s_now = float(obs["road"].get("s", 0.0))
                if s_now >= float(self._road_s_end) - float(self.env_cfg.road_success_s_margin_m):
                    terminated = True
                    terminal_reason = "success"
                    success_via = "road_end"
            except Exception:
                pass
        if (not terminated) and self.env_cfg.done_on_out_of_bounds and out_of_bounds:
            terminated = True
            terminal_reason = "out_of_bounds"

        # 道路任务：离开路面直接终止（可选）
        if (
            (not terminated)
            and bool(self.env_cfg.done_on_out_of_road)
            and isinstance(obs.get("road"), dict)
            and (not bool(obs["road"].get("in_road", True)))
        ):
            terminated = True
            terminal_reason = "out_of_road"

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
            road_s=float(obs.get("road", {}).get("s", 0.0)) if isinstance(obs.get("road"), dict) else None,
            prev_road_s=float(self._prev_road_s) if self._prev_road_s is not None else None,
            road_cte=float(obs.get("road", {}).get("cte", 0.0)) if isinstance(obs.get("road"), dict) else None,
            road_heading_error=float(obs.get("road", {}).get("heading_error", 0.0)) if isinstance(obs.get("road"), dict) else None,
            in_road=bool(obs.get("road", {}).get("in_road", True)) if isinstance(obs.get("road"), dict) else None,
        )

        self._prev_action = cmd.astype(np.float32, copy=False)
        self._prev_goal_distance = gd
        if pos_xy is not None:
            self._prev_pos_xy = pos_xy
        if isinstance(obs.get("road"), dict):
            try:
                self._prev_road_s = float(obs["road"].get("s", 0.0))
            except Exception:
                pass
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
            "linear_x_odom": float(obs.get("linear_x", 0.0)),
            "angular_z_odom": float(obs.get("angular_z", 0.0)),
            "oob_position_xy": oob_xy,
            "out_of_bounds": out_of_bounds,
            "terminal_reason": terminal_reason or "",
            "success_via": success_via,
            "collision_via": collision_via,
        }
        if isinstance(obs.get("road"), dict):
            try:
                info["road_s"] = float(obs["road"].get("s", 0.0))
                info["road_cte"] = float(obs["road"].get("cte", 0.0))
                info["in_road"] = bool(obs["road"].get("in_road", True))
            except Exception:
                pass
        # world 对齐诊断：与 Gazebo world 内的 marker(绿/红圆柱)坐标一致（与 step 内 success 共用一次查询）
        if wpose_step is not None:
            wx, wy, wyaw = wpose_step
            info["world_position_xy"] = (float(wx), float(wy))
            info["world_yaw"] = float(wyaw)
            gx_w = float(self.env_cfg.goal_x)
            gy_w = float(self.env_cfg.goal_y)
            info["world_goal_xy"] = (gx_w, gy_w)
            if world_goal_d is not None:
                info["world_goal_distance"] = float(world_goal_d)
            else:
                info["world_goal_distance"] = float(np.hypot(gx_w - float(wx), gy_w - float(wy)))

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
        if self._obstacle_mgr is not None:
            try:
                self._obstacle_mgr.clear_spawned()
            except Exception:
                pass
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
