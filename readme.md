# 采用PPO训练的多模态无人避障小车项目

## 项目简介

## 系统架构

## 运行流程

## 训练流程

## 实验结果

## 快速开始（推荐只记一个入口）

安装依赖：

```bash
pip install -r requirements.txt
```

Mock 训练（不依赖 ROS/Gazebo）：

```bash
python3 run.py train --task mock --model ppo --total-updates 50 --rollout-steps 512
```

Gazebo 训练（需已启动仿真并 `source ros2_ws/install/setup.bash`）：

```bash
python3 run.py train --task gazebo --model ppo --total-updates 20 --rollout-steps 256
```

Gazebo 链路冒烟测试（订阅 scan/odom/camera 并发布 cmd_vel）：

```bash
python3 run.py smoke
```

## 目录约定（精简版）

- `obstacle_environment/`：强化学习环境（ROS2 话题订阅、观测/动作/奖励/重置等）
- `rl_algorithms/`：PPO 实现与训练入口
- `ros2_ws/src/`：ROS2 包源码（仅保留 `src`；`build/ install/ log/` 属于生成物，已忽略）
- `run.py`：唯一通用入口（配置 task/model 并运行）
- `maps/`：Gazebo world 地图素材（如需可自行加入版本控制或按需下载）
- `logs/ checkpoints/`：训练产物（运行时生成，默认忽略）

阶段 0：准备环境

安装 ROS2（Humble 或 Iron）

安装 Gazebo 或 Ignition Gazebo

安装 Python 包（torch, gym, numpy, opencv, stable-baselines3 等）

测试 ROS2 + Gazebo 是否能启动 TurtleBot3（或你自己的小车）

目标：确保仿真环境可以运行，ROS2节点可以发布/订阅 topic。

阶段 1：搭建 ROS2 仿真
任务顺序：

rl_car_description

写小车的 URDF/xacro 文件

定义雷达、相机、车体

测试 ros2 launch 能够 spawn 机器人

rl_car_gazabo

创建仿真 world

放置障碍物

创建 launch 文件，让小车 spawn 到 world

rl_car_sensors

创建雷达节点：发布 /scan

创建相机节点：发布 /camera/image

测试 ros2 topic echo 是否有数据

rl_car_control

创建一个简单控制节点，订阅 /cmd_vel，控制小车

测试小车能动

rl_car_brigde

把 ROS2 topic 转化成 Python 数据

例如：LaserScan → numpy array

Image → tensor

做成一个 Gym environment接口

阶段 2：构建强化学习环境（obstacle_environment）

gym_env

写 step(), reset(), render()

step() 接收动作 → 发布 /cmd_vel → 获取状态 → 计算奖励

reset() 重置 Gazebo 小车位置和障碍

observation

定义状态向量：

state = [lidar, velocity, goal_distance, goal_angle]

如果加相机：CNN特征 + lidar

action

定义动作空间：

连续动作：[linear_vel, angular_vel]

离散动作：前进 / 左转 / 右转 / 停止

reward

定义奖励函数：

避障成功：正奖励

碰撞：负奖励

前进奖励：正奖励，鼓励小车移动

reset

随机生成障碍物

随机小车初始位置

保证训练多样性

阶段 3：填充强化学习算法（rl_algorithms）

已实现（见仓库 `rl_algorithms/`）：

- `rl_algorithms/ppo/`：`ActorCritic`（MLP）、`PPOTrainer`（GAE + clip）
- `rl_algorithms/envs/mock_car_env.py`：与 `state_dim=18` 一致的玩具环境，**无 ROS 可跑通训练**
- `rl_algorithms/train_ppo.py`：命令行训练入口

**依赖**：`pip install -r requirements.txt`（建议项目内 venv：`python3 -m venv .venv && source .venv/bin/activate`）

**一键脚本（推荐）**：默认 **启动带窗口的 Gazebo** 再训练，便于观察小车；超参数在 `scripts/train_ppo_launch.sh` 顶部修改。

```bash
cd graduation_project
./scripts/train_ppo_launch.sh
```

- 仿真日志：`logs/ppo_gazebo_train_sim.log`
- 已有仿真在跑时：在脚本里设 `START_GAZEBO=0`
- 无显示器时：`GAZEBO_GUI=false`
- 仅离线测试：`MODE=mock`

**冒烟训练（Mock，命令行直调）**：

```bash
cd graduation_project
python -m rl_algorithms.train_ppo --mock --total-updates 50 --rollout-steps 512
# 权重默认保存到 checkpoints/ppo_car.pt
```

**Gazebo 训练**（需已 `ros2 launch rl_car_gazebo sim.launch.py` 并 `source ros2_ws/install/setup.bash`）：

```bash
python -m rl_algorithms.train_ppo --gazebo --total-updates 20 --rollout-steps 256
```

说明：仿真步较慢时可减小 `--rollout-steps` 与 `--total-updates` 先验证链路。

阶段 4：实验与评估

evaluation

加载训练好的模型

在不同场景下测试

输出指标：碰撞率、路径长度、成功率

logs

保存训练曲线

TensorBoard 或 matplotlib

阶段 5：配置和脚本

config

写 YAML 文件：

PPO参数（学习率、gamma、步长）

环境参数（障碍数量、最大速度）

奖励权重

scripts

train.sh：训练脚本

test.sh：测试脚本

launch_sim.sh：启动 ROS2 + Gazebo + PPO桥接

阶段 6：文档和论文素材

docs/ 存架构图、实验图、论文草稿

readme.md 说明系统流程、运行方法

填充顺序总结（优先级）

ROS2 + Gazebo基础能跑

Gym环境接口

PPO算法接口

训练循环

实验评估

文档和论文图表

⚡ Tip：先能让小车在仿真中动起来，比写复杂的PPO网络重要。
PPO训练前，你必须确保 ROS2小车 + 障碍环境 + 状态动作奖励 都能工作。