# rl_algorithms

## 结构

| 路径 | 说明 |
|------|------|
| `ppo/networks.py` | `ActorCritic`：连续动作，高斯经 tanh 压到 `(-1,1)` |
| `ppo/ppo.py` | `PPOConfig`、`PPOTrainer`（rollout、GAE、多 epoch minibatch 更新） |
| `envs/mock_car_env.py` | 无 ROS 的 18 维状态玩具环境 |
| `train_ppo.py` | `python -m rl_algorithms.train_ppo` |

## 运行

```bash
# 项目根目录，建议 venv
pip install -r requirements.txt
python -m rl_algorithms.train_ppo --mock --total-updates 30
```

加载权重示例：

```python
import torch
from rl_algorithms.ppo.networks import ActorCritic

ckpt = torch.load("checkpoints/ppo_car.pt", map_location="cpu")
net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
net.load_state_dict(ckpt["net"])
```
