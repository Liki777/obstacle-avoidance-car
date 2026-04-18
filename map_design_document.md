# PPO 动态避障小车：随机障碍物与 Level1~3 训练设计文档

## 1. 总体目标

目标是在 ROS2 + Gazebo + TurtleBot3 环境中，使用 PPO 算法训练小车实现动态避障。

整个系统分为三部分：

1. Gazebo 障碍物生成
2. PPO 状态-动作-奖励设计
3. 分阶段训练（Level1~Level3）

推荐采用课程式训练（Curriculum Learning）：

* Level1：先学静态障碍物
* Level2：再加入少量动态障碍物
* Level3：最后加入多个动态障碍物

这样可以降低 PPO 训练难度，避免一开始环境过于复杂导致策略无法收敛。

---

# 2. 随机障碍物生成规则

## 2.1 地图区域定义

先定义障碍物允许出现的区域：

```python
x_min, x_max = -8.0, 8.0
y_min, y_max = -8.0, 8.0
```

地图中还需要定义：

```python
robot_start = (0.0, 0.0)
goal_pos = (6.0, 6.0)
```

---

## 2.2 障碍物生成限制

为了保证训练稳定，随机障碍物不能完全无约束生成。

建议加入以下规则：

1. 障碍物不能离机器人起点太近
2. 障碍物不能离目标点太近
3. 障碍物之间不能重叠
4. 障碍物不能堵死所有道路
5. 每个障碍物需要有唯一名称

推荐阈值：

```python
min_dist_to_robot = 1.5
min_dist_to_goal = 1.5
min_dist_between_obstacles = 1.0
```

---

## 2.3 随机位置生成伪代码

```python
obstacles = []

for i in range(num_obstacles):
    valid = False

    while not valid:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        # 检查与机器人起点距离
        if distance((x, y), robot_start) < min_dist_to_robot:
            continue

        # 检查与目标点距离
        if distance((x, y), goal_pos) < min_dist_to_goal:
            continue

        # 检查与已有障碍物距离
        overlap = False
        for obs in obstacles:
            if distance((x, y), obs.position) < min_dist_between_obstacles:
                overlap = True
                break

        if overlap:
            continue

        valid = True

    obstacles.append({
        "name": f"obstacle_{i}",
        "x": x,
        "y": y
    })
```

---

## 2.4 Gazebo 中生成障碍物伪代码

```python
for obstacle in obstacles:
    spawn_model(
        name=obstacle["name"],
        sdf_path="box/model.sdf",
        x=obstacle["x"],
        y=obstacle["y"],
        z=0.0
    )
```

生成障碍物通常使用 Gazebo 服务：

```python
/gazebo/spawn_sdf_model
```

---

# 3. 动态障碍物实现规则

动态障碍物的核心是不断更新模型位置。

一般流程：

1. 先随机生成障碍物初始位置
2. 给每个障碍物分配运动模式
3. 定时调用 set_model_state 更新位置

常用 Gazebo 服务：

```python
/gazebo/set_model_state
```

---

## 3.1 左右移动障碍物

```python
x = x0 + A * sin(t)
y = y0
```

伪代码：

```python
for each timestep:
    obstacle.x = obstacle.x0 + amplitude * sin(current_time)
    obstacle.y = obstacle.y0
```

---

## 3.2 上下移动障碍物

```python
x = x0
y = y0 + A * sin(t)
```

---

## 3.3 圆周运动障碍物

```python
x = x0 + r * cos(t)
y = y0 + r * sin(t)
```

---

## 3.4 随机游走障碍物

随机游走障碍物每隔固定时间改变方向：

```python
if step % 50 == 0:
    theta = random.uniform(-pi, pi)

x += speed * cos(theta)
y += speed * sin(theta)
```

---

# 5. Level1：随机静态障碍物

## 5.1 训练目标

让小车学会：

* 不撞障碍物
* 能够绕开障碍物
* 能到达目标点

---

## 5.2 环境设置

```python
num_static_obstacles = 3~5
num_dynamic_obstacles = 0
```

障碍物只生成一次，不会移动。

推荐地图：

* 空旷区域
* 障碍物分散
* 保留明显通路

---

## 5.3 Level1 伪代码

```python
reset_environment()
spawn_random_static_obstacles()

while not done:
    state = get_state()
    action = ppo.predict(state)
    apply_action(action)

    next_state = get_state()
    reward = compute_reward()

    store_transition(state, action, reward, next_state)
```

---

# 6. Level2：随机静态 + 单个动态障碍物

## 6.1 训练目标

让小车学会：

* 避开静态障碍物
* 识别动态障碍物运动趋势
* 等待障碍物离开后再前进

---

## 6.2 环境设置

```python
num_static_obstacles = 4~6
num_dynamic_obstacles = 1
```

动态障碍物推荐采用简单运动：

* 左右移动
* 上下移动

---

## 6.3 Level2 伪代码

```python
reset_environment()
spawn_random_static_obstacles()
spawn_one_dynamic_obstacle()

while not done:
    update_dynamic_obstacle_position()

    state = get_state()
    action = ppo.predict(state)
    apply_action(action)

    next_state = get_state()
    reward = compute_reward()

    store_transition(state, action, reward, next_state)
```

---

# 7. Level3：随机静态 + 多动态障碍物

## 7.1 训练目标

让小车学会：

* 同时处理多个动态目标
* 在复杂场景中寻找安全路径
* 提前预测动态障碍物未来位置

---

## 7.2 环境设置

```python
num_static_obstacles = 5~8
num_dynamic_obstacles = 2~4
```

动态障碍物可使用不同运动模式混合：

* 左右移动
* 上下移动
* 圆周运动
* 随机游走

---

## 7.3 Level3 伪代码

```python
reset_environment()
spawn_random_static_obstacles()
spawn_multiple_dynamic_obstacles()

while not done:
    for obstacle in dynamic_obstacles:
        update_obstacle_motion(obstacle)

    state = get_state()
    action = ppo.predict(state)
    apply_action(action)

    next_state = get_state()
    reward = compute_reward()

    store_transition(state, action, reward, next_state)
```

---

# 8. 推荐训练流程

建议训练顺序：

```python
train_level1()
load_best_model()
train_level2()
load_best_model()
train_level3()
```

这样 PPO 会逐步适应更复杂的环境。

推荐每个阶段至少训练：

* Level1：100k ~ 300k steps
* Level2：300k ~ 600k steps
* Level3：600k+ steps

---


