"""
最终版奖励范式（工程可落地 + PPO 稳定收敛）。

R =
  + k_progress * progress_raw（默认：位移在「指向目标」方向的投影；可关 use_displacement_progress 回退距离差）
  + k_direction * cos(goal_angle)（默认 1.0 即纯 cos）- k_behind_goal * max(0, |angle|-pi/2)
  + k_velocity_direction *   v * cos(goal_angle)
  - k_front_close_penalty * max(0, margin_m - lidar_front_min)
  + k_lateral_detour * 1[|world_y| > lateral_detour_thresh_m]
  + k_safe     * safe_distance(min_lidar)
  - k_risk     * (|v| / min_lidar)
  - k_turn     * |w|
  - k_stop     * 1[|v| < v_min]
  - k_smooth   * ||a_t - a_{t-1}||_1

终止项（由 env 触发）：
  collision -> -collision_penalty
  success   -> +success_reward
  out_of_bounds -> -out_of_bounds_penalty
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from obstacle_environment.reward.reward_config import RewardConfig


def _finite_non_saturating(
    lidar_ranges: np.ndarray,
    *,
    range_max: float | None,
) -> np.ndarray:
    """
    取有限测距，并剔除 LaserScan 中「满量程 / 无回波」读数（通常 ≈ range_max），
    以便 min 表示真实障碍距离而非最远可见点。
    """
    r = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return finite
    if range_max is None:
        return finite
    hi = float(range_max)
    eps = max(5e-4, 0.002 * hi)
    thr = hi - eps
    close = finite[finite < thr]
    return close


def lidar_min_range(
    lidar_ranges: np.ndarray,
    *,
    range_max: float | None = None,
) -> float:
    """
    一维激光上的「最近障碍物」距离估计 (m)，用于碰撞判定与奖励。

    原先实现：无有限读数时固定返回 1e3，导致 ``dmin <= collision_distance`` 永远不成立，
    贴墙/传感器全无效时 **不会触发 collision done**。

    现逻辑：
    - 若有非满量程的有限回波，取最小值（含略小于 ``range_min`` 的「过近」读数，贴墙常见）。
    - 若全部为 inf/nan：返回 0，视为无可靠测距，env 可触发碰撞终止。
    - 若有限但均为满量程：返回 ``range_max``（开阔）；未提供 ``range_max`` 时退回 ``np.min(finite)``。
    """
    r = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return 0.0

    close = _finite_non_saturating(r, range_max=range_max)
    if close.size > 0:
        return float(np.min(close))

    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return 0.0
    if range_max is not None:
        return float(range_max)
    return float(np.min(finite))


def lidar_front_min_range(
    lidar_ranges: np.ndarray,
    *,
    angle_min: float,
    angle_increment: float,
    half_width_rad: float,
    range_max: float | None = None,
) -> float:
    """
    车头前向扇区（|angle|<=half_width）内的最小有限距离。
    角度与 LaserScan 一致（通常 base_link 前方约 0 rad）。
    """
    r = np.asarray(lidar_ranges, dtype=np.float64).reshape(-1)
    n = int(r.size)
    if n == 0:
        return lidar_min_range(r, range_max=range_max)
    idx = np.arange(n, dtype=np.float64)
    angles = float(angle_min) + idx * float(angle_increment)
    # 归一化到 [-pi, pi] 后比较与 0 的夹角
    a = np.arctan2(np.sin(angles), np.cos(angles))
    mask = np.abs(a) <= float(half_width_rad)
    sector = r[mask]
    close = _finite_non_saturating(sector, range_max=range_max)
    if close.size > 0:
        return float(np.min(close))
    return lidar_min_range(r, range_max=range_max)


@dataclass
class RewardBreakdown:
    """分项与总和，便于 TensorBoard / CSV 记录。"""

    total: float
    progress: float
    direction: float
    safe: float
    risk: float
    turn: float
    stop: float
    smooth: float
    front_safe: float = 0.0
    front_risk: float = 0.0
    terminal: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = {
            "reward_total": float(self.total),
            "reward_progress": float(self.progress),
            "reward_direction": float(self.direction),
            "reward_safe": float(self.safe),
            "reward_risk": float(self.risk),
            "reward_front_safe": float(self.front_safe),
            "reward_front_risk": float(self.front_risk),
            "reward_turn": float(self.turn),
            "reward_stop": float(self.stop),
            "reward_smooth": float(self.smooth),
            "reward_terminal": float(self.terminal),
        }
        d.update({k: float(v) for k, v in self.components.items()})
        return d


def safe_distance_reward(min_lidar: float, cfg: RewardConfig) -> float:
    """f(d)=clip((d-d_min)/(d_safe-d_min),0,1)"""
    d = float(min_lidar)
    d0 = float(cfg.collision_distance)
    d1 = float(cfg.safe_distance)
    if d1 <= d0:
        return 0.0
    v = (d - d0) / (d1 - d0)
    return float(np.clip(v, 0.0, 1.0))


def progress_reward(goal_distance: float, prev_goal_distance: Optional[float]) -> float:
    if prev_goal_distance is None:
        return 0.0
    return float(prev_goal_distance) - float(goal_distance)


def displacement_progress_toward_goal(
    prev_xy: tuple[float, float],
    curr_xy: tuple[float, float],
    goal_x: float,
    goal_y: float,
) -> float:
    """
    世界系下位移 (prev→curr) 在「curr→goal」单位向量上的投影。
    朝目标移动为正，切向/绕圈为接近 0，远离为负。
    """
    dx = float(curr_xy[0] - prev_xy[0])
    dy = float(curr_xy[1] - prev_xy[1])
    to_gx = float(goal_x - curr_xy[0])
    to_gy = float(goal_y - curr_xy[1])
    dist = float(np.hypot(to_gx, to_gy))
    if dist < 1e-6:
        return 0.0
    ux, uy = to_gx / dist, to_gy / dist
    return float(dx * ux + dy * uy)


def _axis_band_overlap_depth_m(*, x_robot: float, x_line: float, half_width_m: float) -> float:
    """
    将「线」近似为以 x_line 为中心、半宽 half_width_m 的带状区域；
    返回车体 x 与该带重叠的深度（米），0 表示完全不在带内。
    """
    hw = max(float(half_width_m), 0.0)
    if hw <= 0.0:
        return 0.0
    d = abs(float(x_robot) - float(x_line))
    return float(max(0.0, hw - d))


def compute_reward(
    *,
    lidar_ranges: np.ndarray,
    goal_distance: float,
    prev_goal_distance: Optional[float] = None,
    goal_angle: float = 0.0,
    linear_x: float = 0.0,
    angular_z: float = 0.0,
    lidar_front_min: Optional[float] = None,
    lidar_range_max: Optional[float] = None,
    prev_robot_xy: Optional[tuple[float, float]] = None,
    robot_xy: Optional[tuple[float, float]] = None,
    goal_x: float = 0.0,
    goal_y: float = 0.0,
    action: Optional[Sequence[float]] = None,
    prev_action: Optional[Sequence[float]] = None,
    config: Optional[RewardConfig] = None,
    terminal: Optional[str] = None,
    road_s: Optional[float] = None,
    prev_road_s: Optional[float] = None,
    road_cte: Optional[float] = None,
    road_heading_error: Optional[float] = None,
    in_road: Optional[bool] = None,
) -> RewardBreakdown:
    """
    Args:
        lidar_ranges: 原始或下采样前的一维距离数组
        linear_x: 车体前向速度 (m/s)，与 ROS base_link 前向一致时取 twist.linear.x
        goal_distance: 当前到目标欧氏距离
        prev_goal_distance: 上一时刻距离；在 ``use_displacement_progress=False`` 时用距离差作 progress
        prev_robot_xy / robot_xy / goal_x / goal_y: 位移投影型 progress（与 odom 同坐标系）
    """
    cfg = config or RewardConfig()
    d_min = lidar_min_range(lidar_ranges, range_max=lidar_range_max)
    d_front = float(lidar_front_min) if lidar_front_min is not None else d_min
    v = float(linear_x)
    w = float(angular_z)
    theta = float(goal_angle)

    # 终止覆盖（由 env 传入终止原因）
    if terminal == "collision":
        return RewardBreakdown(
            total=-float(cfg.collision_penalty),
            progress=0.0,
            direction=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            front_safe=0.0,
            front_risk=0.0,
            terminal=-float(cfg.collision_penalty),
            components={"lidar_min_m": d_min, "lidar_front_min_m": d_front, "terminal": 1.0},
        )
    if terminal == "success":
        return RewardBreakdown(
            total=float(cfg.success_reward),
            progress=0.0,
            direction=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            front_safe=0.0,
            front_risk=0.0,
            terminal=float(cfg.success_reward),
            components={"lidar_min_m": d_min, "lidar_front_min_m": d_front, "terminal": 1.0},
        )
    if terminal == "out_of_bounds":
        return RewardBreakdown(
            total=-float(cfg.out_of_bounds_penalty),
            progress=0.0,
            direction=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            front_safe=0.0,
            front_risk=0.0,
            terminal=-float(cfg.out_of_bounds_penalty),
            components={"lidar_min_m": d_min, "lidar_front_min_m": d_front, "terminal": 1.0},
        )
    if terminal == "out_of_road":
        return RewardBreakdown(
            total=-float(cfg.out_of_road_penalty),
            progress=0.0,
            direction=0.0,
            safe=0.0,
            risk=0.0,
            turn=0.0,
            stop=0.0,
            smooth=0.0,
            front_safe=0.0,
            front_risk=0.0,
            terminal=-float(cfg.out_of_road_penalty),
            components={
                "lidar_min_m": d_min,
                "lidar_front_min_m": d_front,
                "terminal": 1.0,
                "in_road": 0.0 if (in_road is False) else 1.0,
            },
        )

    # ---- road-following terms (optional) ----
    r_road_prog = 0.0
    r_road_cte = 0.0
    r_road_heading = 0.0
    if road_s is not None and prev_road_s is not None and float(cfg.k_road_progress) != 0.0:
        ds = float(road_s) - float(prev_road_s)
        r_road_prog = float(cfg.k_road_progress) * float(ds)
    if road_cte is not None and float(cfg.k_road_cte) != 0.0:
        r_road_cte = -float(cfg.k_road_cte) * abs(float(road_cte))
    if road_heading_error is not None and float(cfg.k_road_heading) != 0.0:
        r_road_heading = -float(cfg.k_road_heading) * abs(float(road_heading_error))

    # ---- lane marking soft penalties (world-x bands; matches straight road worlds) ----
    r_lane = 0.0
    dep_yellow = 0.0
    dep_white_l = 0.0
    dep_white_r = 0.0
    dep_div_l = 0.0
    dep_div_r = 0.0
    if robot_xy is not None:
        xr = float(robot_xy[0])
        if float(cfg.k_road_yellow_line) != 0.0:
            hw_y = float(cfg.road_yellow_half_width_m)
            dep_yellow = _axis_band_overlap_depth_m(
                x_robot=xr, x_line=float(cfg.road_yellow_line_x_a), half_width_m=hw_y
            ) + _axis_band_overlap_depth_m(
                x_robot=xr, x_line=float(cfg.road_yellow_line_x_b), half_width_m=hw_y
            )
            r_lane -= float(cfg.k_road_yellow_line) * float(dep_yellow)
        if float(cfg.k_road_white_line) != 0.0:
            dep_white_l = _axis_band_overlap_depth_m(
                x_robot=xr,
                x_line=float(cfg.road_white_edge_left_x),
                half_width_m=float(cfg.road_white_half_width_m),
            )
            dep_white_r = _axis_band_overlap_depth_m(
                x_robot=xr,
                x_line=float(cfg.road_white_edge_right_x),
                half_width_m=float(cfg.road_white_half_width_m),
            )
            r_lane -= float(cfg.k_road_white_line) * float(max(dep_white_l, dep_white_r))
        if float(cfg.k_road_lane_divider) != 0.0:
            dep_div_l = _axis_band_overlap_depth_m(
                x_robot=xr,
                x_line=float(cfg.road_lane_divider_left_x),
                half_width_m=float(cfg.road_lane_divider_half_width_m),
            )
            dep_div_r = _axis_band_overlap_depth_m(
                x_robot=xr,
                x_line=float(cfg.road_lane_divider_right_x),
                half_width_m=float(cfg.road_lane_divider_half_width_m),
            )
            r_lane -= float(cfg.k_road_lane_divider) * float(max(dep_div_l, dep_div_r))

    r_reverse = 0.0
    if float(cfg.k_reverse) != 0.0:
        veps = max(float(cfg.reverse_v_eps), 0.0)
        if v < -veps:
            r_reverse = -float(cfg.k_reverse) * float(-v - veps)

    if (
        cfg.use_displacement_progress
        and prev_robot_xy is not None
        and robot_xy is not None
    ):
        raw_prog = displacement_progress_toward_goal(
            prev_robot_xy, robot_xy, float(goal_x), float(goal_y)
        )
    else:
        raw_prog = progress_reward(goal_distance, prev_goal_distance)

    prog_scale = 1.0
    if abs(theta) > float(cfg.progress_heading_gate_rad):
        prog_scale = float(cfg.progress_heading_gate_scale)
    r_progress = float(cfg.k_progress) * float(raw_prog) * prog_scale

    cos_t = float(np.cos(theta))
    r_dir = float(cfg.k_direction) * cos_t
    behind_excess = max(0.0, abs(float(theta)) - (math.pi / 2.0))
    if behind_excess > 0.0:
        r_dir -= float(cfg.k_behind_goal) * behind_excess
    # 动作方向一致性：朝目标方向前进奖励更大，反向前进惩罚
    r_vdir = float(cfg.k_velocity_direction) * (v * cos_t)
    r_safe = cfg.k_safe * safe_distance_reward(d_min, cfg)

    denom = max(float(d_min), float(cfg.risk_eps))
    r_risk = -cfg.k_risk * (abs(v) / denom)

    r_front_safe = cfg.k_front_safe * safe_distance_reward(d_front, cfg)
    denom_f = max(float(d_front), float(cfg.risk_eps))
    r_front_risk = -cfg.k_front_risk * (abs(v) / denom_f)

    margin_fc = float(cfg.front_close_penalty_margin_m)
    r_front_close = 0.0
    if float(d_front) < margin_fc:
        r_front_close = -float(cfg.k_front_close_penalty) * (margin_fc - float(d_front))

    r_detour = 0.0
    if robot_xy is not None and float(cfg.k_lateral_detour) != 0.0:
        if abs(float(robot_xy[1])) > float(cfg.lateral_detour_thresh_m):
            r_detour = float(cfg.k_lateral_detour)

    r_turn = -cfg.k_turn * abs(w)

    r_stop = -cfg.k_stop if abs(v) < float(cfg.v_min) else 0.0

    r_smooth = 0.0
    if action is not None and prev_action is not None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        p = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        n = min(int(a.size), int(p.size))
        if n > 0:
            r_smooth = -cfg.k_smooth * float(np.sum(np.abs(a[:n] - p[:n])))

    total = float(
        r_progress
        + r_dir
        + r_vdir
        + r_safe
        + r_risk
        + r_front_safe
        + r_front_risk
        + r_front_close
        + r_detour
        + r_road_prog
        + r_road_cte
        + r_road_heading
        + r_lane
        + r_reverse
        + r_turn
        + r_stop
        + r_smooth
    )

    return RewardBreakdown(
        total=total,
        progress=float(r_progress),
        direction=float(r_dir),
        safe=float(r_safe),
        risk=float(r_risk),
        turn=float(r_turn),
        stop=float(r_stop),
        smooth=float(r_smooth),
        front_safe=float(r_front_safe),
        front_risk=float(r_front_risk),
        terminal=0.0,
        components={
            "lidar_min_m": float(d_min),
            "lidar_front_min_m": float(d_front),
            "goal_distance": float(goal_distance),
            "goal_angle": float(theta),
            "progress_raw": float(raw_prog),
            "progress_scale_heading_gate": float(prog_scale),
            "safe_unit": safe_distance_reward(d_min, cfg),
            "v": v,
            "w": w,
            "cos_theta": cos_t,
            "r_dir": float(r_dir),
            "r_vdir": float(r_vdir),
            "reward_front_close": float(r_front_close),
            "reward_detour": float(r_detour),
            "road_s": float(road_s) if road_s is not None else 0.0,
            "road_cte": float(road_cte) if road_cte is not None else 0.0,
            "road_heading_error": float(road_heading_error) if road_heading_error is not None else 0.0,
            "reward_road_progress": float(r_road_prog),
            "reward_road_cte": float(r_road_cte),
            "reward_road_heading": float(r_road_heading),
            "reward_lane_markings": float(r_lane),
            "lane_dep_yellow_m": float(dep_yellow),
            "lane_dep_white_max_m": float(max(dep_white_l, dep_white_r)),
            "lane_dep_divider_max_m": float(max(dep_div_l, dep_div_r)),
            "reward_reverse": float(r_reverse),
            "in_road": 0.0 if (in_road is False) else 1.0,
        },
    )


def compute_reward_from_observation(
    obs: Mapping[str, Any],
    *,
    lidar_ranges: np.ndarray,
    prev_goal_distance: Optional[float] = None,
    config: Optional[RewardConfig] = None,
    **kwargs: Any,
) -> RewardBreakdown:
    """
    与 `build_observation` 返回的 dict 配合使用。

    - 优先使用 ``obs["linear_x"]``（车体前向速度）作为前进奖励；
      若无则回退到 ``velocity``（合速度模长）。
    - 激光最小距需传入原始 ``lidar_ranges``（一维，与 LaserScan.ranges 一致）。
    """
    lx = obs.get("linear_x", obs.get("velocity", 0.0))
    wz = obs.get("angular_z", 0.0)
    gd = float(obs.get("goal_distance", 0.0))
    ga = float(obs.get("goal_angle", 0.0))
    return compute_reward(
        lidar_ranges=lidar_ranges,
        linear_x=float(lx),
        angular_z=float(wz),
        goal_distance=gd,
        goal_angle=ga,
        prev_goal_distance=prev_goal_distance,
        config=config,
        **kwargs,
    )
