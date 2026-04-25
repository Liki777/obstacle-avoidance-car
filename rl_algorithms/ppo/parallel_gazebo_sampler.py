"""
多进程并行采样（Gazebo/ROS2）。

设计目标：在 **主进程仅做 PPO update** 的同时，让多个子进程各自连接一套隔离的 Gazebo+ROS 图
（通过 ROS_DOMAIN_ID + GAZEBO_MASTER_URI）并行 collect_rollout，从而显著提高采样吞吐。

约束/假设：
- 每个 worker 进程内会 `rclpy.init()` 并创建 `RlCarGazeboEnv`
- 每个 worker 负责启动/关闭自己的 Gazebo（ros2 launch rl_car_gazebo sim.launch.py ...）
- 主进程不初始化 rclpy（避免 domain/graph 干扰）
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Mapping, Optional

import numpy as np
import torch

from rl_algorithms.ppo.networks import ActorCritic


@dataclass(frozen=True)
class ParallelGazeboConfig:
    num_envs: int = 4
    base_ros_domain_id: int = 10
    base_gazebo_master_port: int = 11345
    world: str = "level1_arena_8x8.world"
    gui: bool = False
    server: bool = True
    use_sim_time: bool = True
    log_dir: str = ""
    # 启动后等待 Gazebo/话题就绪的秒数（worker 内仍会 reset->wait_ready 做兜底）
    launch_settle_sec: float = 7.0


def _make_env_vars(*, ros_domain_id: int, gazebo_master_port: int) -> dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(int(ros_domain_id))
    env["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{int(gazebo_master_port)}"
    # 禁用在线模型库访问（WSL/无网络时常导致 libcurl 报错并拖慢/退出 gzclient）
    env.setdefault("GAZEBO_MODEL_DATABASE_URI", "")
    # 抑制无声卡环境 OpenAL 噪声
    env.setdefault("ALSOFT_DRIVERS", "null")
    return env


def _launch_gazebo(
    *,
    env: dict[str, str],
    world: str,
    gui: bool,
    server: bool,
    use_sim_time: bool,
    stdout_f,
) -> subprocess.Popen:
    # 通过 ros2 launch 复用项目内 sim.launch.py（会启动 gzserver、可选 gzclient、并 spawn 机器人）
    cmd = [
        "ros2",
        "launch",
        "rl_car_gazebo",
        "sim.launch.py",
        f"world:={world}",
        f"gui:={'true' if gui else 'false'}",
        f"server:={'true' if server else 'false'}",
        f"use_sim_time:={'true' if use_sim_time else 'false'}",
    ]
    return subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=subprocess.STDOUT, text=True)


def _worker_main(
    *,
    wid: int,
    conn,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    device: str,
    rollout_steps: int,
    env_cfg_kwargs: dict[str, Any],
    spec_snapshot: Optional[dict[str, Any]],
    pcfg: ParallelGazeboConfig,
) -> None:
    # isolate ROS/Gazebo graph
    ros_domain = int(pcfg.base_ros_domain_id) + int(wid)
    gz_port = int(pcfg.base_gazebo_master_port) + int(wid)
    env_vars = _make_env_vars(ros_domain_id=ros_domain, gazebo_master_port=gz_port)
    # 关键：worker 进程自身也必须使用同一 ROS_DOMAIN_ID，否则看不到 Gazebo 的 topic/service
    # （仅给 subprocess(ros2 launch) 传 env 不够，因为 rclpy 读的是当前进程环境变量）
    os.environ.update(env_vars)

    proc: subprocess.Popen | None = None
    log_f = None
    try:
        log_dir = str(pcfg.log_dir).strip()
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"worker_{wid}.log")
            log_f = open(log_path, "a", encoding="utf-8")
            log_f.write(
                f"\n=== worker {wid} start | ROS_DOMAIN_ID={ros_domain} "
                f"GAZEBO_MASTER_URI={env_vars.get('GAZEBO_MASTER_URI','')} world={pcfg.world} ===\n"
            )
            log_f.flush()

        def _emit(line: str) -> None:
            s = str(line).rstrip("\n")
            if not s:
                return
            if log_f is not None:
                try:
                    log_f.write(s + "\n")
                    log_f.flush()
                except Exception:
                    pass
            try:
                # 只发送纯文本，主进程会加 [workerX] 前缀
                conn.send({"type": "log", "wid": int(wid), "line": s})
            except Exception:
                pass

        _emit(f"[worker{wid}] launching gazebo...")
        proc = _launch_gazebo(
            env=env_vars,
            world=str(pcfg.world),
            gui=bool(pcfg.gui),
            server=bool(pcfg.server),
            use_sim_time=bool(pcfg.use_sim_time),
            stdout_f=(log_f if log_f is not None else subprocess.DEVNULL),
        )
        time.sleep(float(pcfg.launch_settle_sec))
        _emit(f"[worker{wid}] gazebo launch settle done ({pcfg.launch_settle_sec:.1f}s)")

        # 延迟 import，避免主进程不必要引入 rclpy
        from obstacle_environment.gym_env.gym_env import GazeboEnvConfig, RlCarGazeboEnv
        from obstacle_environment.robot_spec import RobotTaskSpec

        cfg = GazeboEnvConfig(**env_cfg_kwargs)
        if isinstance(spec_snapshot, dict) and spec_snapshot:
            spec = RobotTaskSpec.from_snapshot_dict(spec_snapshot)
        else:
            # 兼容旧路径：仅按 road_map 打开 include_road（奖励/动作边界可能与主进程不一致）
            use_road = bool(str(getattr(cfg, "road_map_yaml", "")).strip())
            spec = RobotTaskSpec.preset_diff_drive(
                include_road=use_road,
                road_lookahead_n=int(getattr(cfg, "road_lookahead_n", 5)),
            )
        env = RlCarGazeboEnv(spec, cfg)
        # 主动跑一次 reset/ready，避免主进程第一轮 rollout 卡在 “env 还没收到 /scan /odom”
        try:
            if hasattr(env, "spin_ros"):
                env.spin_ros(240)
            _ = env.reset()
        except Exception as e:
            _emit(f"[worker{wid}] env warmup reset failed: {e}")
            raise
        _emit(f"[worker{wid}] env ready; waiting for rollout requests")

        net = ActorCritic(obs_dim, act_dim, hidden_dim=hidden_dim).to(torch.device(device))
        net.eval()

        while True:
            msg = conn.recv()
            if not isinstance(msg, dict):
                continue
            typ = msg.get("type")
            if typ == "close":
                break
            if typ == "set_weights":
                sd = msg.get("state_dict")
                if isinstance(sd, dict):
                    net.load_state_dict(sd)
                continue
            if typ != "rollout":
                continue

            # rollout (基本复制 PPOTrainer.collect_rollout，但不做 update)
            _emit(f"[worker{wid}] rollout start steps={int(rollout_steps)}")
            obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
            act_buf = np.zeros((rollout_steps, act_dim), dtype=np.float32)
            logp_buf = np.zeros((rollout_steps,), dtype=np.float32)
            rew_buf = np.zeros((rollout_steps,), dtype=np.float32)
            done_buf = np.zeros((rollout_steps,), dtype=np.float32)
            val_buf = np.zeros((rollout_steps + 1,), dtype=np.float32)

            if hasattr(env, "spin_ros"):
                env.spin_ros(180)
            o = env.reset()

            ep_done = 0
            ep_lens: list[int] = []
            cur_ep_len = 0
            ep_ret = 0.0
            term_counts: dict[str, int] = {"collision": 0, "success": 0, "out_of_bounds": 0, "truncated": 0, "other": 0}

            with torch.no_grad():
                for t in range(int(rollout_steps)):
                    ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    action, logp, v = net.act(ot)
                    a = action.cpu().numpy().reshape(-1)

                    obs_buf[t] = o
                    act_buf[t] = a
                    logp_buf[t] = float(logp.cpu().numpy().item())
                    val_buf[t] = float(v.cpu().numpy().item())

                    o, r, done, info = env.step(a)
                    rew_buf[t] = float(r)
                    done_buf[t] = float(done)
                    cur_ep_len += 1
                    ep_ret += float(r)
                    if done:
                        reason = ""
                        terminated = False
                        truncated = False
                        if isinstance(info, dict):
                            reason = str(info.get("terminal_reason", "") or "")
                            terminated = bool(info.get("terminated", False))
                            truncated = bool(info.get("truncated", False))
                        disp = reason or ("truncated" if truncated else "other")
                        if disp in term_counts:
                            term_counts[disp] += 1
                        elif truncated:
                            term_counts["truncated"] += 1
                        else:
                            term_counts["other"] += 1
                        if log_f is not None:
                            try:
                                log_f.write(
                                    f"[episode] return={float(ep_ret):.4f} steps={int(cur_ep_len)} "
                                    f"reason={disp} terminated={terminated} truncated={truncated}\n"
                                )
                            except Exception:
                                pass
                        _emit(
                            f"[worker{wid}] [episode] return={float(ep_ret):.4f} steps={int(cur_ep_len)} "
                            f"reason={disp} terminated={terminated} truncated={truncated}"
                        )
                        ep_done += 1
                        ep_lens.append(int(cur_ep_len))
                        cur_ep_len = 0
                        ep_ret = 0.0
                        o = env.reset()

                ot = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, v_last = net.act(ot)
                val_buf[int(rollout_steps)] = float(v_last.cpu().numpy().item())

            # GAE（与 PPOTrainer._compute_gae 等价）
            adv = np.zeros((rollout_steps,), dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(int(rollout_steps))):
                next_nonterminal = 1.0 - float(done_buf[t])
                next_val = float(val_buf[t + 1])
                delta = float(rew_buf[t]) + float(msg["gamma"]) * next_val * next_nonterminal - float(val_buf[t])
                last_gae = delta + float(msg["gamma"]) * float(msg["gae_lambda"]) * next_nonterminal * last_gae
                adv[t] = float(last_gae)
            ret = adv + val_buf[:-1]
            adv = (adv - float(np.mean(adv))) / (float(np.std(adv)) + 1e-8)

            mean_step_r = float(np.mean(rew_buf)) if int(rollout_steps) > 0 else 0.0
            mean_abs_step_r = float(np.mean(np.abs(rew_buf))) if int(rollout_steps) > 0 else 0.0
            _emit(
                f"[worker{wid}] rollout done | mean_step_r={mean_step_r:.4f} "
                f"episodes={int(ep_done)} term={term_counts}"
            )
            conn.send(
                {
                    "type": "rollout",
                    "obs": obs_buf,
                    "actions": act_buf,
                    "logprobs": logp_buf,
                    "advantages": adv.astype(np.float32),
                    "returns": ret.astype(np.float32),
                    "values": val_buf[:-1].astype(np.float32),
                    "episodes_done": int(ep_done),
                    "mean_ep_len": float(sum(ep_lens) / max(1, len(ep_lens))) if ep_lens else 0.0,
                    "mean_step_reward": mean_step_r,
                    "mean_abs_step_reward": mean_abs_step_r,
                    "term_collision": int(term_counts["collision"]),
                    "term_success": int(term_counts["success"]),
                    "term_oob": int(term_counts["out_of_bounds"]),
                    "term_trunc": int(term_counts["truncated"]),
                    "term_other": int(term_counts["other"]),
                }
            )
    except BaseException as e:
        try:
            _emit(f"[worker{wid}] ERROR: {e}")
            conn.send({"type": "error", "error": str(e)})
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        if log_f is not None:
            try:
                log_f.write(f"=== worker {wid} exit ===\n")
                log_f.flush()
                log_f.close()
            except Exception:
                pass


class ParallelGazeboSampler:
    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        device: str,
        rollout_steps: int,
        env_cfg_kwargs: dict[str, Any],
        spec_snapshot: Optional[Mapping[str, Any]] = None,
        pcfg: ParallelGazeboConfig,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = str(device)
        self.rollout_steps = int(rollout_steps)
        self.env_cfg_kwargs = dict(env_cfg_kwargs)
        self.spec_snapshot = dict(spec_snapshot) if isinstance(spec_snapshot, Mapping) else None
        self.pcfg = pcfg

        # 关键：使用 spawn，避免继承主进程已 init 的 rclpy/ROS_DOMAIN，导致 worker 看不到自己 domain 的 Gazebo 服务
        self._mp_ctx = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._conns: list[Any] = []

    def start(self) -> None:
        n = int(self.pcfg.num_envs)
        for wid in range(n):
            parent, child = self._mp_ctx.Pipe(duplex=True)
            p = self._mp_ctx.Process(
                target=_worker_main,
                kwargs=dict(
                    wid=wid,
                    conn=child,
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                    rollout_steps=self.rollout_steps,
                    env_cfg_kwargs=self.env_cfg_kwargs,
                    spec_snapshot=self.spec_snapshot,
                    pcfg=self.pcfg,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._conns.append(parent)

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send({"type": "close"})
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass

    def set_weights(self, state_dict: dict[str, Any]) -> None:
        for c in self._conns:
            c.send({"type": "set_weights", "state_dict": state_dict})

    def collect(self, *, gamma: float, gae_lambda: float) -> list[dict[str, Any]]:
        # 先广播 rollout 请求
        for c in self._conns:
            c.send({"type": "rollout", "gamma": float(gamma), "gae_lambda": float(gae_lambda)})

        # 并行等待：持续 drain 各 worker 的 log；每个 worker 收到首个 type=rollout 的响应后标记完成
        pending = set(range(len(self._conns)))
        out: list[dict[str, Any]] = [None for _ in range(len(self._conns))]  # type: ignore[list-item]

        while pending:
            progressed = False
            for i in list(pending):
                c = self._conns[i]
                if not c.poll(0.05):
                    continue
                msg = c.recv()
                progressed = True
                if isinstance(msg, dict) and msg.get("type") == "log":
                    wid = int(msg.get("wid", i))
                    line = str(msg.get("line", ""))
                    # worker 端有些日志行已带 "[workerX]" 前缀；避免重复输出成 "[workerX] [workerX] ..."
                    s = line.lstrip()
                    if s.startswith("[worker"):
                        print(line, flush=True)
                    else:
                        print(f"[worker{wid}] {line}", flush=True)
                    continue
                out[i] = msg
                pending.remove(i)
            if not progressed:
                # 防止 tight loop 吃满 CPU
                time.sleep(0.02)

        return out  # type: ignore[return-value]

