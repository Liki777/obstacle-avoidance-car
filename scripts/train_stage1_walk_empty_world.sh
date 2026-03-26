#!/usr/bin/env bash
# =============================================================================
# 第一阶段：empty.world + 随机目标（每个 episode 重置一次） => 让模型先“学会走路/导航基本盘”
#
# 目标：
# - 地图极简（ground plane），减少探索难度
# - 每回合随机目标，让策略必须“朝目标走”，而不是记住一个固定点
# - reward 不引入雷达相关惩罚（k_safe/k_risk=0），但 state 仍保留 lidar 输入，保证后续阶段无缝迁移
#
# 运行：
#   chmod +x scripts/train_stage1_walk_empty_world.sh
#   ./scripts/train_stage1_walk_empty_world.sh
#
# 输出：
# - 训练日志：终端实时打印
# - 步级 CSV：logs/stage1_steps.csv（每步 action/观测/reward 分项）
# - 仿真日志：logs/stage1_sim.log
# =============================================================================

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ckpt_utils.sh"

# ---------- 可配置区 ----------
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROOT}/ros2_ws"

# Gazebo 可视化
GAZEBO_GUI="${GAZEBO_GUI:-true}"
WORLD="empty.world"
SIM_LOG="${ROOT}/logs/stage1_sim.log"

# 训练超参（先小一点，保证很快看到 update 输出）
TOTAL_UPDATES="${TOTAL_UPDATES:-80}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
CONTROL_DT="${CONTROL_DT:-0.05}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"

# 目标随机范围（empty.world 没边界墙，给一个合理训练场地）
# 建议范围不宜太大，否则 early stage 进展信号很弱
GOAL_RANGE_X="${GOAL_RANGE_X:--4.0,4.0}"
GOAL_RANGE_Y="${GOAL_RANGE_Y:--4.0,4.0}"
GOAL_MIN_DISTANCE="${GOAL_MIN_DISTANCE:-1.2}"

# 出界判定（empty.world 没墙，必须靠出界终止防止跑飞）
MAP_X_MIN="${MAP_X_MIN:--6.0}"
MAP_X_MAX="${MAP_X_MAX:-6.0}"
MAP_Y_MIN="${MAP_Y_MIN:--6.0}"
MAP_Y_MAX="${MAP_Y_MAX:-6.0}"

# 出生点与朝向
SPAWN_X="${SPAWN_X:-0.0}"
SPAWN_Y="${SPAWN_Y:-0.0}"
SPAWN_YAW="${SPAWN_YAW:-0.0}"

# 保存与步日志
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/stage1}"
# 默认：stage1 自动从 stage0/latest.pt 起步（若存在），否则从本 stage 的 latest.pt 起步
PREV_CKPT_DIR="${PREV_CKPT_DIR:-${ROOT}/checkpoints/stage0}"
DEFAULT_LOAD="$(latest_ckpt "${PREV_CKPT_DIR}" 2>/dev/null || true)"
LOAD_PATH="${LOAD_PATH:-${DEFAULT_LOAD:-$(latest_ckpt "${CKPT_DIR}" 2>/dev/null || true)}}"
SAVE_PATH="${SAVE_PATH:-$(next_ckpt_path "${CKPT_DIR}" checkpoint)}"
STEP_LOG_CSV="${STEP_LOG_CSV:-${ROOT}/logs/stage1_steps.csv}"

mkdir -p "${ROOT}/logs" "${ROOT}/checkpoints"

# ---------- 清理：确保 Gazebo 能关干净 ----------
SIM_PID=""
cleanup() {
  echo ""
  echo "[stage1] cleaning up..."
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill -TERM -- "-${SIM_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---------- source ROS + workspace ----------
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${WS}/install/setup.bash"
set -u

# 如果你用项目内 venv，把 env_obstacle 改成你的 venv 名称
if [[ -f "${ROOT}/env_obstacle/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env_obstacle/bin/activate"
fi

echo "[stage1] launching Gazebo (gui=${GAZEBO_GUI}) world=${WORLD}"
rm -f "${SIM_LOG}"

LAUNCH_FILE="${WS}/src/rl_car_gazebo/launch/sim.launch.py"
setsid bash -lc "
  set +u
  source \"/opt/ros/${ROS_DISTRO}/setup.bash\"
  source \"${WS}/install/setup.bash\"
  exec ros2 launch \"${LAUNCH_FILE}\" \
    \"world:=${WORLD}\" \
    \"gui:=${GAZEBO_GUI}\" \
    \"x:=${SPAWN_X}\" \
    \"y:=${SPAWN_Y}\" \
    \"z:=0.1\" \
    \"yaw:=${SPAWN_YAW}\" \
    >\"${SIM_LOG}\" 2>&1
" &
SIM_PID=$!
echo "[stage1] sim pid=${SIM_PID} log=${SIM_LOG}"

echo "[stage1] waiting for /scan and /odom..."
TLIST() { ros2 topic list -t 2>/dev/null || true; }
for _ in $(seq 1 240); do
  if TLIST | grep -qE '^/scan ' && TLIST | grep -qE '^/odom '; then
    break
  fi
  sleep 0.5
done

echo "[stage1] start training (reward_profile=stage1_walk, random goals each reset)"
cd "${ROOT}"
python3 -m rl_algorithms.train_ppo \
  --gazebo \
  --reward-profile stage1_walk \
  ${LOAD_PATH:+--load "${LOAD_PATH}"} \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --total-updates "${TOTAL_UPDATES}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --control-dt "${CONTROL_DT}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --spawn-x "${SPAWN_X}" \
  --spawn-y "${SPAWN_Y}" \
  --spawn-yaw "${SPAWN_YAW}" \
  --goal-range-x="${GOAL_RANGE_X}" \
  --goal-range-y="${GOAL_RANGE_Y}" \
  --goal-min-distance "${GOAL_MIN_DISTANCE}" \
  --map-x-min "${MAP_X_MIN}" \
  --map-x-max "${MAP_X_MAX}" \
  --map-y-min "${MAP_Y_MIN}" \
  --map-y-max "${MAP_Y_MAX}" \
  --step-log-csv "${STEP_LOG_CSV}" \
  --save "${SAVE_PATH}"

update_latest_link "${SAVE_PATH}"

