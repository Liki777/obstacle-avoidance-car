# 项目文件说明（维护索引）

本文件用于解释仓库内每个目录/关键文件的用途与边界，避免“文件越堆越多、放哪都不对”的失控状态。
后续**每次新增/修改文件**，都需要在这里补一条说明（新增：写用途/输入输出/被谁调用；修改：写行为变化与影响面）。

## 入口

- `run.py`
  - **用途**：项目唯一通用入口（训练/冒烟测试），通过 `--task` 与 `--model` 选择任务与算法。
  - **边界**：只做参数路由与最薄的一层 glue，不写复杂业务逻辑。
  - **注意**：
    - `train` 使用独立解析：`python3 run.py train ...` 会先解析常用参数（如 `--level`），再把剩余未知参数透传给 `rl_algorithms.train_ppo`。
    - 透传参数开头多余的 `--` 会自动去掉，避免 `run.py train ... -- --level 0` 触发 argparse 的选项结束符语义导致报错。
- `show_map.py`
  - **用途**：通用地图查看脚本：`python3 show_map.py [map]` 启动 `rl_car_gazebo` 的 `sim.launch.py` 展示对应 world。
  - **约定**：map 的 `.world` 文件应放在 `ros2_ws/src/rl_car_gazebo/worlds/`，不要堆到 `maps/`。
  - **注意**：新增 world 后需要 `cd ros2_ws && colcon build && source install/setup.bash`，因为 `sim.launch.py` 从 install 的 package share 读取 world。

## 强化学习（算法侧）

- `rl_algorithms/`
  - **用途**：PPO 训练实现（网络、rollout、update）与训练 CLI。
  - **关键文件**：
    - `rl_algorithms/train_ppo.py`：PPO 训练入口（被 `run.py train` 调用）。
      - 约定：使用 `--level N` 时默认保存到 `checkpoints/levelN/latest.pt`；且 level>0 默认自动加载 `checkpoints/level{N-1}/latest.pt`（除非显式 `--load` 或 `--no-auto-prev-load`）。
      - Level0：若仍使用默认 maze 边界参数，会自动放宽 `map_x/y_*`，避免在空旷 world 上频繁 `out_of_bounds` 造成“不断重启”。
      - `--oob-pose-frame`：控制越界判定用的坐标系；`world` 用 TF 将 `base_link` 变换到 `world`，更贴近 Gazebo 视觉上的“离开地图区域”。留空时 level0 默认 `world`，其它 level 默认 `odom`（兼容旧 maze 训练逻辑）。
      - Level1+：`--level>=1` 时会在 `GazeboEnvConfig` 打开 `enable_level1_static_obstacles`（每回合 reset 随机静态障碍）。
    - `rl_algorithms/ppo/ppo.py`：PPOTrainer（rollout + GAE + clip 更新）。
      - `collect_rollout()` 会汇总 reward 分项均值、终止原因计数、episode 长度等诊断信息，供 `train_ppo` 打印定位训练异常。
    - `rl_algorithms/ppo/networks.py`：ActorCritic 网络结构。
    - `rl_algorithms/envs/mock_car_env.py`：不依赖 ROS 的 mock 环境（快速自检/调参用）。

## 环境（任务侧）

- `obstacle_environment/`
  - **用途**：Gazebo/ROS2 下的环境封装（观测、动作映射、奖励、reset 等）。
  - **关键文件**：
    - `obstacle_environment/gym_env/gym_env.py`：`RlCarGazeboEnv`（reset/step 的主流程）。
      - 碰撞终止：仅在激光存在“可信近距离回波”时才启用，避免 `lidar_min_range()==0` 的哨兵值导致第 0 步误报 collision。
      - 线速度 lidar 缩放：当上一步 `d_front` 近似 0（不可靠）时不缩放，避免线速度被压到接近 0 造成原地转圈。
      - 越界：`map_x/y_*` 默认按训练区定义；若你以 Gazebo world 观察“离开地图”，应使用 `GazeboEnvConfig.oob_pose_frame="world"`（或训练时 `--oob-pose-frame world`）。
    - `obstacle_environment/observation/`：状态向量构建（lidar/odom/goal 等）。
    - `obstacle_environment/action/`：动作空间与 cmd_vel 映射/裁剪。
    - `obstacle_environment/reward/`：奖励配置与计算（`reward_config.py` / `reward_computer.py`）。
    - `obstacle_environment/world_generator/static_obstacles.py`：Level1 静态障碍采样（纯数据/不调用 Gazebo）。
    - `obstacle_environment/scenario_manager/gazebo_obstacle_manager.py`：Gazebo 障碍物 spawn/delete 管理器（有服务则生效，无服务自动降级）。

## ROS2 仿真工程

- `ros2_ws/src/`
  - **用途**：ROS2 包源码（Gazebo 启动、机器人描述、传感器/控制等）。
  - **约定**：`ros2_ws/build/ install/ log/` 属于生成物，已被 `.gitignore` 忽略，不进入版本控制。
  - **关键资源**：
    - `ros2_ws/src/rl_car_gazebo/worlds/level0_map.world`：Level0 走路底图（起点绿圈、终点红圈，固定目标）。
    - `ros2_ws/src/rl_car_gazebo/worlds/level1_map.world`：Level1 空旷底图（障碍由训练时动态 spawn）。
      - 说明：`<gravity>` 放在 `<world>` 级别，避免 Gazebo Classic 对 physics 参数解析报错。

## 地图与素材

- `maps/`
  - **用途**：Gazebo `.world` 等“静态素材”。
  - **边界**：只放**可复用、相对稳定**的 world/模型资源；不要把“随机生成逻辑/课程逻辑/训练脚本”塞进这里。

## 文档

- `map_design_document.md`
  - **用途**：训练地图/障碍生成的需求说明（Level1~3、静态/动态规则）。
  - **说明**：实现以“生成器模块 + 场景管理器”方式落地，避免把逻辑写进 `.world` 文件本身。
