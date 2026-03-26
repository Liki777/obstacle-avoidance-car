#!/usr/bin/env bash
# =============================================================================
# Stage0：empty.world + 固定目标点
#
# 目的：
# - 最基础导航阶段：先学会“从出生点走向一个固定目标”
# - 关闭雷达惩罚（reward_profile=stage1_walk），但 state 保留 lidar 输入
# - 不随机目标，降低任务复杂度，快速验证策略能否收敛到“朝目标走”
#
# 运行：
#   chmod +x scripts/train_stage0_fixed_goal.sh
#   ./scripts/train_stage0_fixed_goal.sh
#
# 输出：
# - 仿真日志：logs/stage0_sim.log
# - 步级日志：logs/stage0_steps.csv
# - 模型：checkpoints/ppo_stage0_fixed_goal.pt
# =============================================================================

set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ckpt_utils.sh"

# ---------- 可配置区 ----------
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROOT}/ros2_ws"

GAZEBO_GUI="${GAZEBO_GUI:-true}"
WORLD="empty.world"
SIM_LOG="${ROOT}/logs/stage0_sim.log"

# 训练超参（stage0 建议更短回合 + 更快迭代）
TOTAL_UPDATES="${TOTAL_UPDATES:-60}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
CONTROL_DT="${CONTROL_DT:-0.05}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-180}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"

# 固定目标（核心）
GOAL_X="${GOAL_X:-3.0}"
GOAL_Y="${GOAL_Y:-0.0}"

# 出生点/朝向
SPAWN_X="${SPAWN_X:-0.0}"
SPAWN_Y="${SPAWN_Y:-0.0}"
SPAWN_YAW="${SPAWN_YAW:-0.0}"

# 训练边界（empty.world 无墙，用边界避免跑飞）
MAP_X_MIN="${MAP_X_MIN:--6.0}"
MAP_X_MAX="${MAP_X_MAX:-6.0}"
MAP_Y_MIN="${MAP_Y_MIN:--6.0}"
MAP_Y_MAX="${MAP_Y_MAX:-6.0}"

CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/stage0}"
SAVE_PATH="${SAVE_PATH:-$(next_ckpt_path "${CKPT_DIR}" checkpoint)}"
LOAD_PATH="${LOAD_PATH:-$(latest_ckpt "${CKPT_DIR}" 2>/dev/null || true)}"
STEP_LOG_CSV="${STEP_LOG_CSV:-${ROOT}/logs/stage0_steps.csv}"

mkdir -p "${ROOT}/logs" "${ROOT}/checkpoints"

SIM_PID=""
cleanup() {
  echo ""
  echo "[stage0] cleaning up..."
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill -TERM -- "-${SIM_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${WS}/install/setup.bash"
set -u

if [[ -f "${ROOT}/env_obstacle/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env_obstacle/bin/activate"
fi

echo "[stage0] launching Gazebo (gui=${GAZEBO_GUI}) world=${WORLD}"
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
echo "[stage0] sim pid=${SIM_PID} log=${SIM_LOG}"

echo "[stage0] waiting for /scan and /odom..."
TLIST() { ros2 topic list -t 2>/dev/null || true; }
for _ in $(seq 1 240); do
  if TLIST | grep -qE '^/scan ' && TLIST | grep -qE '^/odom '; then
    break
  fi
  sleep 0.5
done

echo "[stage0] start training (fixed goal: ${GOAL_X}, ${GOAL_Y})"
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
  --goal-x "${GOAL_X}" \
  --goal-y "${GOAL_Y}" \
  --map-x-min "${MAP_X_MIN}" \
  --map-x-max "${MAP_X_MAX}" \
  --map-y-min "${MAP_Y_MIN}" \
  --map-y-max "${MAP_Y_MAX}" \
  --step-log-csv "${STEP_LOG_CSV}" \
  --save "${SAVE_PATH}"

update_latest_link "${SAVE_PATH}"

