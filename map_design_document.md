# PPO 小车避障：课程 Level0~5 与地图设计说明

## 1. 总体目标与结论

**目标**：在 ROS2 + Gazebo 环境下，用 PPO 学到从起点安全到达终点的策略；随课程逐步提高**静态几何不确定性**与**动态障碍不确定性**，最终具备在「随机静态 + 随机动态」并存场景中的泛化能力。

**课程结论（是否采用当前六级划分）**：

- **可以采用**。顺序符合「感知运动规律 → 空间约束 → 引入时间维度 → 再叠加组合难度」的课程学习思路，有利于 PPO 稳定收敛。
- **设计要点**：Level3 使用**固定轨迹**动态体，便于策略先学会「时序预测 + 让行」；Level4 再引入**随机参数**动态体，降低过拟合到单一正弦/圆周模式的风险；Level5 与最终目标最接近。
- **可选增强（非必须）**：在后期训练中混入少量早期样本（例如 Level5 时 5%~10% 仅用静态或仅用动态），可减轻对某一子任务的遗忘；若训练时间有限可不做。

---

## 2. 课程 Level0 ~ Level5 定义

| Level | 静态障碍 | 动态障碍 | 训练侧重点 |
|------:|----------|----------|------------|
| **0** | 无 | 无 | 仅学到达目标（速度/朝向/进度奖励 shaping） |
| **1** | **固定**（每局相同布局） | 无 | 绕障几何、可复现调试策略与奖励 |
| **2** | **随机**（每局 reset 重采样） | 无 | 对静态布局泛化 |
| **3** | 无 | **固定**（每局相同运动律：正弦/圆周等） | 预测运动趋势、让行时机 |
| **4** | 无 | **随机**（每局随机初相、振幅、角频率、模式） | 对动态参数泛化 |
| **5** | **随机** | **随机** | 接近最终目标：联合静动态避障 |

**实现约定（与代码一致）**：

- `rl_algorithms/train_ppo.py --level N`（`N∈[0,5]`）自动配置 `GazeboEnvConfig` 的 `static_obstacle_mode` / `dynamic_obstacle_mode` 及默认奖励预设。
- Level1~5 默认启用 **8×8 m** 训练域（越界与采样边界约 `x,y∈[-0.25,8.25]`），默认 **起点 (1,1)**、**终点 (7,7)**；可用 `--no-arena-8x8` 关闭并自行指定 `--map-*` / `--spawn-*` / `--goal-*`。
- Level1 固定静态障碍来自 `builtin_level1_fixed_mixed_3x3()`：场地中部 **3×3 共 9 个**混合体（扁圆柱 / 方柱 / 竖圆柱交替）；亦可通过 `GazeboEnvConfig.fixed_static_obstacles_xyyaw` 自定义为旧版「仅盒列表」布局。
- Level3 固定动态轨迹来自同模块的 `builtin_fixed_dynamic_specs()`；每仿真步用 `SetEntityState` 更新位姿（与文档 3 节一致）。

---

## 3. 动态障碍物实现（仿真层）

动态障碍在 Gazebo Classic 中的实现方式：

1. `SpawnEntity` 生成与静态相同的盒模型（名称前缀 `train_dyn_*`）。
2. 每个控制周期根据解析式更新位姿，调用 **`/set_entity_state`**（或 `/gazebo/set_entity_state`）写入 `world` 系位姿。
3. 角速度设为零，避免与 ODE 积分抢控制权（等价于「运动学动画体」）。

**支持的运动模式（Level3/4/5）**：

- `sin_x`：`x = x0 + A·sin(ωt+φ)`，`y = y0`
- `sin_y`：`y = y0 + A·sin(ωt+φ)`，`x = x0`
- `circle`：`x = x0 + A·cos(ωt+φ)`，`y = y0 + A·sin(ωt+φ)`

Level4/5 在每次 `reset()` 时对 `x0,y0,A,ω,φ` 及模式做均匀/离散随机（需满足不贴墙、不压起点/终点邻域）。

---

## 4. 随机静态障碍物规则（Level2 / Level5）

采样区域与 `GazeboEnvConfig.map_*` 一致；默认 8×8 课程下与围墙内场对齐。

建议约束（代码中 `StaticObstacleSamplingConfig` 可调）：

1. 与机器人起点足够远（默认约 ≥1.5 m）。
2. 与当前目标点足够远。
3. 障碍两两中心距大于阈值（默认约 ≥1.0 m）。
4. 每个障碍唯一命名，便于 `DeleteEntity` 清理。

数量默认：Level2 为 `static_obstacle_count_min/max`（默认 3~5）；Level5 与 Level2 相同逻辑，可与动态数量一起在后续按论文需求再调参。

---

## 5. 推荐地图（Gazebo world）

| Level | 推荐 world（`ros2 launch rl_car_gazebo sim.launch.py world:=...`） | 说明 |
|------:|---------------------------------------------------------------------|------|
| 0 | `level0_arena_8x8.world` 或任意空旷图 | 无障碍；8×8 围墙便于与后续阶段视觉一致 |
| 1 | `level1_arena_8x8.world` | 与固定静态坐标系一致；障碍由程序 spawn |
| 2 | `level1_arena_8x8.world` / `arena_8x8.world` | 与 Level1 同几何，仅采样策略不同 |
| 3 | `level2_arena_8x8.world` 或 `arena_8x8.world` | 无障碍块；动态体由程序生成 |
| 4 | 同上 | 随机动态 |
| 5 | `level3_arena_8x8.world` 或 `arena_8x8.world` | 高对比标记可选；静+动均由程序生成 |

> 旧版 `level1_map.world`（更大训练区）仍保留，用于与历史 checkpoint 对齐；**新课程默认以 8×8 arena 为主**。

---

## 6. 训练流程与 checkpoint

建议顺序：`0 → 1 → 2 → 3 → 4 → 5`，每阶段从上一阶段 `checkpoints/level{N-1}/latest.pt` 热启动（`train_ppo` 默认行为，可用 `--no-auto-prev-load` 关闭）。

示例：

```bash
# 先启动仿真（示例 Level2）
ros2 launch rl_car_gazebo sim.launch.py world:=level1_arena_8x8.world \
  x:=1.0 y:=1.0 z:=0.1 yaw:=0.7854

python -m rl_algorithms.train_ppo --gazebo --level 2 --total-updates 200
```

纯展示（不训练）与开图推荐统一用根目录 ``run.py`` / ``show_map.py``：

- 开 Gazebo 地图：``python3 run.py show --level 1`` 或 ``python3 show_map.py --level 1``（与课程 world 一致）。
- 障碍演示：``python3 run.py demo --level 1`` 或 ``python3 demo_obstacles.py --level 1``（Level1 固定不刷新；Level2 按 ``--period`` 刷新）。
- 训练：``python3 run.py train --task gazebo --level 1``（先另终端 ``run.py show --level 1`` 起仿真）。

**步数建议（量级，可按算力调整）**：

- Level0：先收敛「到点」行为，约 50k~200k 环境步量级。
- Level1~2：静态绕障与泛化，各约 100k~400k。
- Level3~4：动态专项，各约 200k~600k。
- Level5：联合难度，≥400k 或直至评估成功率稳定。

---

## 7. 奖励与策略（与 `RewardConfig` 预设对齐）

默认映射（可用 `--reward-profile` 覆盖）：

- Level0：`stage1_walk`（强调进度与朝向，弱化「只对齐不打分」的投机解）。
- Level1~2：`stage2_simple_avoid`。
- Level3~5：`stage3_hard_avoid`（更强调风险与前方净空）。

策略网络结构不必随 Level 改变；**观测空间维数不变**时同一 MLP 可贯穿课程。若后续加入显式速度通道或预测头，再在文档与代码中单独开版本说明。

---

## 8. 附录：旧版 Level1~3 叙述的迁移说明

早期文档将「Level1=随机静态、Level2=静+单动态、Level3=静+多动态」与当前 **Level0~5** 六级划分**不一致**。以本节为准；代码与 `train_ppo --level` 已与六级课程对齐。
