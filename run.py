#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _normalize_passthrough(passthrough: list[str]) -> list[str]:
    """
    argparse 会把单独的 ``--`` 当作“选项结束符”。
    用户常用 ``run.py train ... -- --level 0`` 这种写法，会把 ``--`` 原样转发给底层 CLI，
    导致后续 ``--level`` 等参数被误解析为位置参数并报 unrecognized arguments。
    """
    out = list(passthrough)
    while out and out[0] == "--":
        out.pop(0)
    return out


def _strip_conflicting_flags(argv: list[str], *, conflict_flags: set[str]) -> list[str]:
    """
    从 argv 中移除与 conflict_flags 冲突的 ``--flag``（以及紧随其后的一个参数值，若存在）。
    用于：run.py 先解析常用参数并显式转发，同时允许用户继续透传其它 train_ppo 参数。
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in conflict_flags:
            # skip flag
            i += 1
            # skip one value token if it doesn't look like another flag
            if i < len(argv) and not str(argv[i]).startswith("-"):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _run_train(*, model: str, task: str, passthrough: list[str]) -> int:
    """
    通用训练入口：
    - model: 目前仅支持 ppo
    - task: mock/gazebo
    - passthrough: 额外参数原样转发给底层 trainer 的 CLI（目前复用 rl_algorithms.train_ppo）
    """
    model = (model or "").strip().lower()
    task = (task or "").strip().lower()
    if model != "ppo":
        raise SystemExit(f"暂不支持 model={model!r}（目前仅支持 'ppo'）")
    if task not in ("mock", "gazebo"):
        raise SystemExit(f"暂不支持 task={task!r}（请选择 'mock' 或 'gazebo'）")

    # 复用现有 argparse（rl_algorithms/train_ppo.py 的 main() 读取 sys.argv）
    from rl_algorithms import train_ppo

    argv = [f"--{task}", *_normalize_passthrough(passthrough)]
    old = sys.argv[:]
    try:
        sys.argv = ["python -m rl_algorithms.train_ppo", *argv]
        train_ppo.main()
        return 0
    finally:
        sys.argv = old


def _run_eval(*, model: str, task: str, passthrough: list[str]) -> int:
    """加载 checkpoint，在环境中跑若干回合（见 train_ppo 的 --eval-episodes）。"""
    model = (model or "").strip().lower()
    task = (task or "").strip().lower()
    if model != "ppo":
        raise SystemExit(f"暂不支持 model={model!r}（目前仅支持 'ppo'）")
    if task not in ("mock", "gazebo"):
        raise SystemExit(f"暂不支持 task={task!r}（请选择 'mock' 或 'gazebo'）")

    from rl_algorithms import train_ppo

    argv = [f"--{task}", *_normalize_passthrough(passthrough)]
    old = sys.argv[:]
    try:
        sys.argv = ["python -m rl_algorithms.train_ppo", *argv]
        train_ppo.main()
        return 0
    finally:
        sys.argv = old


def _run_smoke(argv: list[str]) -> int:
    """
    内置冒烟测试：验证 ROS2 topic 连通性并发布 cmd_vel。
    这是 scripts/smoketest.py 的精简替代（避免再依赖 scripts/ 目录）。
    """
    try:
        import time
        import math
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
    except ModuleNotFoundError as e:
        raise SystemExit(f"缺少 ROS2 Python 依赖，无法运行 smoke：{e}")

    if argv and argv[0] in ("-h", "--help"):
        print(
            "用法: python3 run.py smoke [--seconds 8] [--scan-topic /scan] [--odom-topic /odom] [--cmd-vel-topic /cmd_vel]"
        )
        return 0

    ap = argparse.ArgumentParser(prog="run.py smoke", add_help=False)
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--cmd-vel-topic", default="/cmd_vel")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args(argv)

    class _Smoke(Node):
        def __init__(self) -> None:
            super().__init__("rl_car_smoke")
            self._got_scan = 0
            self._got_odom = 0
            self._pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
            self.create_subscription(LaserScan, args.scan_topic, self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, args.odom_topic, self._on_odom, 10)

        def _on_scan(self, _msg: LaserScan) -> None:
            self._got_scan += 1

        def _on_odom(self, _msg: Odometry) -> None:
            self._got_odom += 1

        def publish_cmd(self, t: float) -> None:
            m = Twist()
            phase = int(t // 4.0) % 2
            if phase == 0:
                m.linear.x = 0.2
                m.angular.z = 0.05 * math.sin(t)
            else:
                m.linear.x = 0.0
                m.angular.z = 0.6 + 0.05 * math.sin(t)
            self._pub.publish(m)

    rclpy.init(args=None)
    node = _Smoke()
    t0 = time.time()
    try:
        while rclpy.ok() and (time.time() - t0) < float(args.seconds):
            t = time.time() - t0
            node.publish_cmd(t)
            rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info(f"smoke done: scan={node._got_scan} odom={node._got_odom}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("用法: python3 run.py {train,eval,smoke} ...")

    cmd = sys.argv[1]
    if cmd in ("-h", "--help"):
        print(
            "用法:\n"
            "  python3 run.py train ...\n"
            "  python3 run.py eval ...\n"
            "  python3 run.py smoke ..."
        )
        return 0

    if cmd == "train":
        p_train = argparse.ArgumentParser(prog="run.py train", description="训练入口（转发 rl_algorithms.train_ppo）")
        p_train.add_argument("--model", type=str, default="ppo", help="模型/算法（目前仅 ppo）")
        p_train.add_argument("--task", type=str, default="mock", help="环境（mock 或 gazebo）")
        p_train.add_argument("--level", type=int, default=None, help="课程阶段（转发给 train_ppo）")
        p_train.add_argument("--total-updates", type=int, default=None, help="PPO update 次数（转发）")
        p_train.add_argument("--rollout-steps", type=int, default=None, help="每次 rollout 步数（转发）")
        p_train.add_argument("--device", type=str, default=None, help="cpu/cuda（转发）")
        p_train.add_argument("--save", type=str, default=None, help="checkpoint 保存路径（转发）")
        p_train.add_argument("--load", type=str, default=None, help="checkpoint 加载路径（转发）")
        p_train.add_argument("--auto-resume", action="store_true", help="若 save 已存在则自动加载（转发）")
        p_train.add_argument("--no-auto-prev-load", action="store_true", help="关闭 level 级联自动加载（转发）")
        p_train.add_argument("--seed", type=int, default=None, help="随机种子（转发）")
        p_train.add_argument(
            "--oob-pose-frame",
            type=str,
            default=None,
            help="越界判定坐标系：odom/world（转发给 train_ppo 的 --oob-pose-frame）",
        )

        t_args, unknown = p_train.parse_known_args(sys.argv[2:])

        forward: list[str] = []
        if t_args.level is not None:
            forward += ["--level", str(int(t_args.level))]
        if t_args.total_updates is not None:
            forward += ["--total-updates", str(int(t_args.total_updates))]
        if t_args.rollout_steps is not None:
            forward += ["--rollout-steps", str(int(t_args.rollout_steps))]
        if t_args.device is not None:
            forward += ["--device", str(t_args.device)]
        if t_args.save is not None:
            forward += ["--save", str(t_args.save)]
        if t_args.load is not None:
            forward += ["--load", str(t_args.load)]
        if bool(t_args.auto_resume):
            forward += ["--auto-resume"]
        if bool(t_args.no_auto_prev_load):
            forward += ["--no-auto-prev-load"]
        if t_args.seed is not None:
            forward += ["--seed", str(int(t_args.seed))]
        if t_args.oob_pose_frame:
            forward += ["--oob-pose-frame", str(t_args.oob_pose_frame)]

        conflict = {
            "--level",
            "--total-updates",
            "--rollout-steps",
            "--device",
            "--save",
            "--load",
            "--auto-resume",
            "--no-auto-prev-load",
            "--seed",
            "--oob-pose-frame",
        }
        rest = _strip_conflicting_flags(_normalize_passthrough(unknown), conflict_flags=conflict)
        return _run_train(model=t_args.model, task=t_args.task, passthrough=forward + rest)

    if cmd == "eval":
        p_ev = argparse.ArgumentParser(
            prog="run.py eval",
            description="加载 checkpoint，连续跑若干完整 episode（默认确定性策略），用于看 Gazebo 效果或统计成功率。",
        )
        p_ev.add_argument("--model", type=str, default="ppo", help="目前仅 ppo")
        p_ev.add_argument("--task", type=str, default="gazebo", help="mock 或 gazebo")
        p_ev.add_argument("--load", type=str, default=None, help="权重路径；省略则尝试 checkpoints/level{N}/latest.pt")
        p_ev.add_argument("--level", type=int, default=None, help="课程阶段（与默认 checkpoint 路径一致）")
        p_ev.add_argument("--episodes", type=int, default=10, help="评估回合数")
        p_ev.add_argument("--stochastic", action="store_true", help="评估时随机采样动作（默认关闭）")
        p_ev.add_argument("--device", type=str, default=None, help="cpu/cuda（转发）")
        p_ev.add_argument("--save", type=str, default=None, help="与 --auto-resume 联用（转发）")
        p_ev.add_argument("--auto-resume", action="store_true", help="若无 --load 且 save 上已有权重则加载（转发）")
        p_ev.add_argument("--seed", type=int, default=None, help="随机种子（转发）")
        p_ev.add_argument(
            "--oob-pose-frame",
            type=str,
            default=None,
            help="越界判定坐标系：odom/world（转发）",
        )
        e_args, unknown = p_ev.parse_known_args(sys.argv[2:])

        lev = int(e_args.level) if e_args.level is not None else 0
        forward: list[str] = ["--eval-episodes", str(int(e_args.episodes))]
        if bool(e_args.stochastic):
            forward.append("--eval-stochastic")
        proj = Path(__file__).resolve().parent
        load_added = False
        if e_args.load is not None:
            forward += ["--load", str(e_args.load)]
            load_added = True
        else:
            cand = proj / "checkpoints" / f"level{lev}" / "latest.pt"
            if cand.is_file():
                forward += ["--load", str(cand)]
                load_added = True
        if not load_added and not bool(e_args.auto_resume):
            raise SystemExit(
                "请指定 --load <checkpoint>，或使用 --auto-resume，"
                f"或先训练生成 {proj / 'checkpoints' / f'level{lev}' / 'latest.pt'}"
            )
        forward += ["--level", str(int(lev))]
        if e_args.device is not None:
            forward += ["--device", str(e_args.device)]
        if e_args.save is not None:
            forward += ["--save", str(e_args.save)]
        if bool(e_args.auto_resume):
            forward += ["--auto-resume"]
        if e_args.seed is not None:
            forward += ["--seed", str(int(e_args.seed))]
        if e_args.oob_pose_frame:
            forward += ["--oob-pose-frame", str(e_args.oob_pose_frame)]

        conflict = {
            "--level",
            "--device",
            "--load",
            "--save",
            "--auto-resume",
            "--seed",
            "--oob-pose-frame",
            "--eval-episodes",
            "--eval-stochastic",
        }
        rest = _strip_conflicting_flags(_normalize_passthrough(unknown), conflict_flags=conflict)
        return _run_eval(model=e_args.model, task=e_args.task, passthrough=forward + rest)

    if cmd == "smoke":
        return _run_smoke(sys.argv[2:])

    raise SystemExit(f"未知子命令: {cmd!r}（支持 train / eval / smoke）")


if __name__ == "__main__":
    raise SystemExit(main())

