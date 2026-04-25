## 改造方案（从“随机障碍到点导航” → “道路驾驶 + 通用避障泛化”）

本文档用于把当前项目从“在空旷区域随机生成障碍、朝目标点走”的设定，改造为**道路场景驾驶**：
策略需要“沿道路行驶、保持在路内、避让静态/动态交通参与者”，并在**多种道路布局**上泛化。

核心动机（为什么必须改）：
- **随机地图到点导航**会出现“绕路/抄近道/贴边走”这类非预期捷径；同时 eval 若道路拓扑随机且不可感知，智能体只能“试错找路”，这不符合毕设展示目标。
- **道路任务**把“可行区域”变成显式结构（路缘/护栏/车道），让 reward/obs 具有可泛化的共同语义，减少“靠运气摸索道路”的不可控性。

---

## 1. 总体架构（保留与改造边界）

### 1.1 可保留（不动或小改）
- **PPO 算法与训练框架**：`rl_algorithms/ppo/ppo.py`、`rl_algorithms/train_ppo.py`（含 auto-resume、latest 覆盖保存、并行采样）。
- **Gazebo 并行采样**：`rl_algorithms/ppo/parallel_gazebo_sampler.py`（ROS_DOMAIN_ID + GAZEBO_MASTER_URI 隔离）。
- **Gazebo reset/调试能力**：`obstacle_environment/gym_env/gym_env.py` 中 reset 服务解析、`--debug-reset`。
- **动态障碍的 SetEntityState 驱动框架**：`obstacle_environment/scenario_manager/gazebo_obstacle_manager.py` + 现有 dynamic specs（将扩展运动模式）。
- **地图启动链路**：`show_map.py`、`run.py show`（可继续用 world 文件可视化）。

### 1.2 必须改造（决定泛化成败）
- **环境任务定义**：从“欧式距离到点”改为“沿道路中心线/waypoints 前进（progress）”。
- **奖励函数**：加入“沿路前进、横向偏离、航向误差、路内约束”的核心项；避免“抄近道奖励更高”。
- **观测**：加入道路几何信息（中心线切向、横向误差、前向若干 waypoint 在车体坐标系下的位置）。
- **地图数据（道路拓扑）**：需要道路中心线/边界的结构化描述（YAML/JSON），并与 world 对应。
- **评估协议**：按“道路族”做 train/eval 划分；不再用“完全随机无结构地图”。

---

## 2. 道路表示（新增 RoadMap 配置文件）

### 2.1 新增文件与目录
- 新增目录：`maps/road/`
- 每个道路布局一份配置：`maps/road/road_*.yaml`

### 2.2 YAML 格式（建议）

```yaml
name: road_s_curve_v1
world: level1_road_s_curve_v1.world   # Gazebo world 文件名（show_map 用）
frame: odom                           # 与你当前 goal_frame_id 对齐

spawn:
  x: 1.0
  y: 1.0
  yaw: 0.0

goal:
  x: 7.0
  y: 7.0

road:
  # 车道中心线（按顺序的 waypoints，单位 m）
  centerline_xy:
    - [1.0, 1.0]
    - [2.0, 1.2]
    - [3.2, 2.0]
    - [4.2, 3.3]
    - [5.0, 4.8]
    - [6.1, 6.1]
    - [7.0, 7.0]
  half_width_m: 0.55                  # 车道半宽；用于“是否在路内”与横向误差

curriculum:
  # 可选：该道路可用于哪些 level
  levels: [0, 1, 2]
```

为什么要这样做：
- **waypoints** 把“路怎么走”以结构化形式给环境计算 reward/obs，而不是让策略靠试错从随机障碍推断“哪里是路”。
- `half_width_m` 提供“路内约束”，避免策略贴墙走或抄近道穿越禁行区域。

---

## 3. 环境改造（RlCarGazeboEnv：引入 RoadTask）

### 3.1 目标：从“到点”改为“沿路前进”
当前环境核心变量多围绕 `goal_distance`。改造后应引入：
- `s`：沿中心线累计弧长（progress）
- `cte`：cross-track error（横向偏差）
- `heading_error`：车头与中心线切线夹角误差
- `lookahead_points`：前方 N 个 waypoint 在车体坐标系下的 (x,y)

### 3.2 改动/新增文件清单
- **新增**：`obstacle_environment/road/road_map.py`
  - 解析 YAML
  - 预处理中心线弧长
  - 提供 `project_to_centerline(xy) -> (s, cte, tangent_yaw)`
  - 提供 `sample_lookahead_points(s, n, ds)`
- **修改**：`obstacle_environment/gym_env/gym_env.py`
  - 在 `reset()` 加载/切换 `RoadMap`（按 level 或参数指定）
  - 在 `build_observation()` 或 `step()` 计算 `cte/heading_error/lookahead`
  - 在 `info` 中加入调试字段：`road_s`, `cte`, `heading_error`, `in_road`
- **修改**：`obstacle_environment/robot_spec.py`（如 observation 维度需要扩展）

### 3.3 为什么要这么改
- 避免“捷径”：reward 与 `Δs` 绑定，而不是与“欧式距离缩短”绑定；欧式距离容易鼓励穿越禁行区域。
- 泛化：不同道路布局下，`cte/heading_error` 的语义一致，策略学的是“在路上开车”，不是记忆某张地图的坐标捷径。

---

## 4. 奖励函数改造（核心：路内驾驶 + 避障）

### 4.1 新增 RoadRewardConfig（建议在 reward_config.py 扩展）
新增/扩展奖励项（示意）：
- `r_progress = k_s * clamp(Δs, -s_clip, s_clip)`
- `r_lane = -k_cte * |cte|`
- `r_heading = -k_yaw * |heading_error|`
- `r_out = -k_out * 1[in_road==False]`（或直接 done）
- 保留你已有的：雷达安全/风险、碰撞终止、成功奖励

关键点：
- **progress 用 Δs**（沿路前进），避免“对角线抄近道”。
- 对 `in_road==False` 给强惩罚或直接终止，杜绝绕外圈。

### 4.2 对应文件改动
- 修改：`obstacle_environment/reward/reward_config.py`
  - 新增 profile：`stage0_road_follow`、`stage1_road_static`、`stage2_road_dynamic`
- 修改：`obstacle_environment/reward/reward_fn.py`（如果你把计算逻辑拆出来更清晰）

---

## 5. 观测改造（让策略看见“路”）

### 5.1 推荐观测字段（在现有 lidar + odom 基础上加）
- `cte`（1 维）
- `heading_error`（1 维）
- `lookahead_points`：例如 N=5，每个点 (x,y) → 10 维

总增量：约 12 维，远小于图像，训练稳定。

### 5.2 对应文件改动
- `obstacle_environment/gym_env/gym_env.py`：在 `obs["state"]` 拼接这些特征
- `obstacle_environment/robot_spec.py`：更新 state_dim

为什么：
- 仅靠 LiDAR 很难“知道路的朝向”；加入中心线几何能显著减少“试错摸路”。

---

## 6. 地图与 Gazebo world（Road worlds）

### 6.1 原则
- world 文件提供**可被 LiDAR 感知**的路缘/护栏（墙/围栏模型），让“路内”有物理意义。
- 静态障碍放在路面内或路缘附近（路锥/路障），动态障碍沿路移动或横穿。

### 6.2 新增 world（最小可行集）
- `level0_road_straight.world`：直路/弯道，无障碍（学会沿路走）
- `level1_road_straight_static.world`：少量路锥/围栏（学静态避障）
- `level2_road_random_static.world`：道路拓扑从若干模板采样 + 随机路锥（学泛化）

对应文件：
- `ros2_ws/src/rl_car_gazebo/worlds/*.world`（新增）
- `ros2_ws/src/rl_car_gazebo/setup.py`（把新 world 加入 data_files）
- `maps/road/*.yaml`（与 world 一一对应）

---

## 7. 动态障碍物（交通参与者）改造

### 7.1 运动模式（与道路绑定）
把动态障碍的运动从“固定正弦/圆周”扩展为：
- `along_centerline`：沿中心线方向以速度 v 前进
- `cross_road`：在特定 s 位置横穿
- `oncoming`：对向来车（沿中心线反向）

### 7.2 对应文件改动
- `obstacle_environment/world_generator/dynamic_obstacle_presets.py`：新增 road 相关 presets
- `obstacle_environment/scenario_manager/gazebo_obstacle_manager.py`：复用 SetEntityState 更新位姿

为什么：
- 让“动态避障”更像交通场景，而不是随机乱动。

---

## 8. Level 设计（落地版，避免“试错找路”）

### Level0：道路跟随（无障碍）
- 固定/随机 road 模板（至少 2 条）
- 目标：最大化 Δs、最小化 cte/heading_error

### Level1：道路 + 少量静态障碍
- 在路内/路缘放少量锥桶/路障（固定或轻随机）
- 目标：在保持路内的前提下绕开障碍继续前进

### Level2：道路族泛化 + 随机静态障碍
- 从道路模板集合采样（训练集），障碍位置/数量随机但保证可通行
- eval 用“未见过的道路模板集合”

### Level3~5：逐步引入动态交通参与者（数量与模式递增）

---

## 9. 训练与评估协议（确保“泛化”可证明）

### 9.1 训练集/测试集拆分（必须做）
- 训练道路集合：`road_train_{A,B,C,...}`
- 测试道路集合：`road_eval_{X,Y,Z,...}`（训练从未见）

### 9.2 指标（建议输出到 CSV）
- 完成率（到达终点/完成路程）
- 平均 cte / 最大 cte
- 碰撞率（静态/动态）
- 平均速度、停滞次数（让行）

对应改动：
- `rl_algorithms/ppo/ppo.py` 的 evaluate_episodes：汇总更多 info 字段（cte/in_road）
- `logs/` 输出新的 eval CSV

---

## 10. 实施里程碑（按可交付顺序）

### M1（最小可运行）：Road Level0
- 1 张直路 world + 1 个 road YAML
- 环境能计算 `s/cte/heading_error`
- reward 用 Δs + 偏离惩罚

### M2：Level1 静态避障
- 加路锥/路障固定模板（仍可通行）
- 增加避障 reward/终止项

### M3：Level2 泛化
- 至少 5 条训练道路 + 3 条测试道路
- 随机静态障碍生成（保证通行）

### M4：Level3+ 动态交通
- 1~3 个动态实体沿路/横穿

---

## 11. 对当前代码的具体改造位置（速查表）

- **新增**：`obstacle_environment/road/road_map.py`（道路几何核心）
- **修改**：`obstacle_environment/gym_env/gym_env.py`
  - reset 加载 road
  - step/obs 计算 s/cte/heading_error/lookahead
- **修改**：`obstacle_environment/reward/reward_config.py`（道路 reward profiles）
- **修改**：`obstacle_environment/robot_spec.py`（观测维度）
- **新增**：`maps/road/*.yaml`（道路族配置）
- **新增**：`ros2_ws/src/rl_car_gazebo/worlds/level*_road_*.world`（道路 world）
- **修改**：`ros2_ws/src/rl_car_gazebo/setup.py`（安装新的 world）
- **可选增强**：`rl_algorithms/ppo/ppo.py evaluate_episodes` 输出道路指标

