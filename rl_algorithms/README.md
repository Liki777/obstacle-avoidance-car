# rl_algorithms（本项目 PPO 训练/评估）

本目录提供 **连续动作 PPO（Clip + GAE）** 的最小实现，并与本仓库的 `obstacle_environment/`（Gazebo+ROS2 环境、观测/动作/奖励）对接。

## 项目架构（与本仓库的关系）

- **环境层（核心在 `obstacle_environment/`）**
  - `RlCarGazeboEnv`：与 Gazebo + ROS2 交互（发布 `/cmd_vel`、订阅 `/scan` `/odom` 等），输出 `state: np.ndarray`，返回 `reward/done/info`。
  - `RobotTaskSpec`：统一 `ObservationConfig` / `ActionConfig` / `RewardConfig`，保证训练时维度与边界一致。
- **算法层（本目录 `rl_algorithms/`）**
  - `ppo/networks.py`：`ActorCritic`（Gaussian policy + tanh squash 到 (-1,1)）。
  - `ppo/ppo.py`：`PPOTrainer`（rollout、GAE、minibatch 多 epoch 更新、保存/加载 checkpoint）。
  - `train_ppo.py`：训练入口（Mock 或 Gazebo），负责把 CLI 参数映射到 `RobotTaskSpec`/`GazeboEnvConfig`。
  - `ppo/parallel_gazebo_sampler.py`：多进程并行采样（每个 worker 启一套隔离的 Gazebo/ROS 图）。

训练时的数据流（简化）：

- **ActorCritic(obs)** → action(tanh) → `RlCarGazeboEnv.step(action)` → next_obs, reward, done, info  
- 收集 `rollout_steps` 后 → **PPO update** → 保存 `latest.pt` 与滚动备份 `checkpoint*.pt`

## 算法细节（PPO 实现要点）

### 1) 策略与动作空间

- **策略输出**：高斯分布 `Normal(mean, std)` 采样后做 `tanh`，动作落在 **(-1,1)**。
- **动作语义**：环境侧用 `ActionMapper` 将 **归一化动作**线性映射到物理区间：
  - `linear_x ∈ [linear_x_min, linear_x_max]`
  - `angular_z ∈ [angular_z_min, angular_z_max]`
- 本仓库目前默认 **角速度区间为 ±0.5 rad/s**（等价于 `angular_z = 0.5 * raw_action_1`）。

### 2) Squashed Gaussian 的 log-prob

`ppo/networks.py` 使用 “tanh-squash” 的变量替换（atanh）计算 log-prob，并加上 \(\log(1-a^2)\) 的 Jacobian 修正，避免直接在 tanh 后变量上用 Normal 产生偏差。

### 3) GAE 与优势归一化

- 使用 **GAE(\(\lambda\))** 计算 advantage
- 在 `collect_rollout()` 末尾对 advantage 做标准化：`(adv-mean)/(std+1e-8)`

### 4) PPO-Clip 更新

- **policy loss**：`max(-adv * ratio, -adv * clip(ratio,1±ε))`
- **value loss**：MSE
- **entropy bonus**：`entropy_coef * entropy`（本项目默认 **0.05**，用于增强探索、缓解“只学某个转向”）
- 梯度裁剪：`max_grad_norm`

### 5) ROS2 回调积压处理（Gazebo）

Gazebo 环境中，PPO 的 update 阶段主线程可能长时间不 `spin`，会造成 DDS 回调积压。`PPOTrainer.collect_rollout()` 会在 reset 前检测 `env.spin_ros` 并主动 spin 一段时间，减少 reset 超时/话题“假断线”。

## 目录结构

| 路径 | 说明 |
|------|------|
| `rl_algorithms/train_ppo.py` | 训练入口（Mock/Gazebo），支持 reward_profile、课程 level、goal 采样与保存/续训 |
| `rl_algorithms/ppo/networks.py` | `ActorCritic`（Gaussian + tanh） |
| `rl_algorithms/ppo/ppo.py` | `PPOConfig`、`PPOTrainer`（rollout/GAE/update/save/load） |
| `rl_algorithms/ppo/parallel_gazebo_sampler.py` | 多进程并行采样（隔离 ROS_DOMAIN_ID + GAZEBO_MASTER_URI） |
| `rl_algorithms/envs/mock_car_env.py` | 无 ROS 的 toy env，用于快速验证 PPO pipeline |

## 使用案例（可直接复制运行）

### A. 本地快速自检（不连 ROS）

```bash
cd /home/liki777/graduation_project
pip install -r requirements.txt
python -m rl_algorithms.train_ppo --mock --total-updates 30
```

### B. Gazebo 训练（单环境）

前提：你已 `colcon build` 并能 source `ros2_ws/install/setup.bash`，Gazebo 可正常启动。

```bash
cd /home/liki777/graduation_project
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source ros2_ws/install/setup.bash

python -m rl_algorithms.train_ppo --gazebo \
  --reward-profile stage2_simple_avoid \
  --step-log-csv logs/ppo_steps.csv \
  --total-updates 200
```

常用参数（部分）：
- `--load <pt>`：从 checkpoint 续训
- `--entropy-coef 0.05`：探索强度（默认已是 0.05）
- `--rollout-steps / --ppo-epochs / --minibatch-size`：采样与更新强度
- `--linear-x-max / --angular-z-max / --angular-z-min`：动作物理上限（会写入 `RobotTaskSpec`）

### C. 跑 10 步打印观测 / 检测策略输出（推荐）

仓库提供探针脚本：`scripts/gazebo_obs_probe.py`（可自动 `ros2 launch rl_car_gazebo sim.launch.py` 并等待 `/scan`+`/odom`）。

```bash
cd /home/liki777/graduation_project
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source ros2_ws/install/setup.bash

# 固定动作 [0.5, 0.0]（归一化动作空间）
python3 scripts/gazebo_obs_probe.py --steps 10

# 加载 checkpoint，打印 actor_mean(pre_tanh)、action(tanh)、value
python3 scripts/gazebo_obs_probe.py --steps 10 \
  --checkpoint checkpoints/stage2_curriculum/course_1/latest.pt \
  --deterministic
```

### D. Python 里加载 checkpoint 并评估若干回合

```python
from rl_algorithms.ppo.ppo import PPOTrainer
from obstacle_environment import RlCarGazeboEnv, RobotTaskSpec

spec = RobotTaskSpec.preset_diff_drive()
env = RlCarGazeboEnv(spec)
trainer = PPOTrainer(env, obs_dim=spec.state_dim, act_dim=spec.action_dim, device="cpu")
trainer.load("checkpoints/stage2_curriculum/course_1/latest.pt")

stats = trainer.evaluate_episodes(10, deterministic=True)
print(stats)
```

## Checkpoint 格式（`.pt`）

`PPOTrainer.save()` 保存的关键字段：

- `net`: `state_dict`
- `opt`: `optimizer state_dict`（可能缺失；load 时允许仅加载网络）
- `obs_dim`, `act_dim`
- `global_update`
- `cfg`: PPO 超参快照（gamma、entropy_coef 等）

`train_ppo.py` 中若 `latest.pt` 已存在，会先复制备份为 `checkpoint{N}.pt`（滚动递增）。

## 并行采样（高级）

`ppo/parallel_gazebo_sampler.py` 的思路是：**每个 worker 自己启动一套 Gazebo**，并用
`ROS_DOMAIN_ID` + `GAZEBO_MASTER_URI` 隔离通信图，从而让主进程专注 PPO update。

建议先确保单环境训练稳定后再启用并行采样。  
