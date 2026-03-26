#!/usr/bin/env bash
# =============================================================================
# 一键：启动「带可视化的 Gazebo」+ PPO 训练（可在本文件顶部改超参数）
#
# 用法：
#   chmod +x scripts/train_ppo_launch.sh   # 首次
#   ./scripts/train_ppo_launch.sh
#
# 默认行为（见下方 MODE / START_GAZEBO）：
#   - 自动 ros2 launch 小车世界，**打开 Gazebo 窗口**，便于观察训练中小车运动
#   - 等待 /scan、/odom 等话题就绪后再开始 python 训练
#   - Ctrl+C 或训练正常结束时，会尝试结束本脚本拉起的 Gazebo 进程
#
# 若已在别处启动了仿真，可设 START_GAZEBO=0，仅连接现有 master。
#
# WSL2：需支持图形界面（WSLg 或 X11），否则把 GAZEBO_GUI=false 改为无头仿真。
# =============================================================================

set -euo pipefail

# ROS2 / Conda / venv 的 setup 脚本常引用「可能未设置」的变量；在 set -u 下会报错。
# 用法：只对「source 某文件」这一段临时关闭 nounset。
sourcenoset() {
  set +u
  # shellcheck source=/dev/null
  source "$1"
  set -u
}

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------

# ---------- 运行模式 ----------
# mock   — 不启 Gazebo，离线玩具环境（无可视化）
# gazebo — 真实仿真 + PPO（默认会启动带窗口的 Gazebo，见 START_GAZEBO）
MODE="gazebo"

# ---------- 是否由本脚本启动 Gazebo + 小车 ----------
# 1 — 本脚本后台执行: ros2 launch rl_car_gazebo sim.launch.py（可配 GUI）
# 0 — 假定你已在另一终端 launch 过，本脚本只负责训练
START_GAZEBO="${START_GAZEBO:-1}"

# ---------- Gazebo 启动参数 ----------
# true  — 打开 Gazebo 三维窗口（推荐，便于「看见训练过程」）
# false — 仅 gzserver，无界面（服务器/无显示环境）
GAZEBO_GUI="${GAZEBO_GUI:-true}"
# 世界文件（与 rl_car_gazebo 包内 worlds/ 一致）
WORLD="${WORLD:-maze.world}"
# 仿真日志（后台 launch 的 stdout/stderr）
SIM_LOG_REL="logs/ppo_gazebo_train_sim.log"
# 传感器话题就绪后，再等待几秒：gzserver 注册 SetModelState 往往晚于 /scan
GAZEBO_SERVICE_SETTLE_SEC="${GAZEBO_SERVICE_SETTLE_SEC:-5}"
# 目标服务名（按你当前环境确认）
GAZEBO_GET_MODEL_LIST_SVC="${GAZEBO_GET_MODEL_LIST_SVC:-/get_model_list}"
GAZEBO_SET_MODEL_STATE_SVC="${GAZEBO_SET_MODEL_STATE_SVC:-/gazebo/set_model_state}"

# ---------- 虚拟环境与 ROS ----------
USE_PROJECT_VENV="${USE_PROJECT_VENV:-1}"
SOURCE_ROS="${SOURCE_ROS:-1}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_WS_INSTALL_REL="ros2_ws/install"

# ---------- 训练总控 ----------
SEED=0
TOTAL_UPDATES=50
# 可视化训练建议先用较小 rollout，终端更快有进度输出
ROLLOUT_STEPS=64
SAVE_PATH="checkpoints/ppo_car.pt"
DEVICE="cpu"
# 每步日志（action + 观测 + reward 分项）
STEP_LOG_CSV="logs/ppo_steps.csv"

# ---------- PPO 超参数 ----------
GAMMA="0.99"
GAE_LAMBDA="0.95"
CLIP_COEF="0.2"
VALUE_COEF="0.5"
ENTROPY_COEF="0.01"
MAX_GRAD_NORM="0.5"
LR="3e-4"
PPO_EPOCHS=10
MINIBATCH_SIZE=64
HIDDEN_DIM=256

# ---------- 仿真环境步长（传给 RlCarGazeboEnv）----------
# 可视化调试：0.05~0.1；值越小训练越快但物理噪声更大
CONTROL_DT="0.05"
MAX_EPISODE_STEPS=256

# ---------- maze 地图边界与重置点（防止跑出地图）----------
# 说明：超出边界会在 env.step() 中直接 done，并触发 reset
# maze.world 外墙中心大致为 x∈[0,10], y∈[-1,9]，这里加裕量
MAP_X_MIN="-0.5"
MAP_X_MAX="10.5"
MAP_Y_MIN="-1.5"
MAP_Y_MAX="9.5"
# 把出生点放在迷宫内部，并让朝向“指向迷宫内部”，避免一 reset 就朝外冲
SPAWN_X="1.0"
SPAWN_Y="0.0"
# 如果你观察到“前进反向”（cmd_vel linear.x>0 却往 -x 走），把 yaw 改成 3.14159
SPAWN_YAW="3.14159"
GOAL_X="2.0"
GOAL_Y="2.0"

# ---------------------------------------------------------------------------
# 路径与清理
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

SIM_LOG="${ROOT}/${SIM_LOG_REL}"
mkdir -p "${ROOT}/logs" "${ROOT}/$(dirname "${SAVE_PATH}")"

# 记录本脚本启动的 launch 进程 PID，便于退出时关掉 Gazebo
SIM_PID=""

cleanup() {
  echo ""
  echo "[train_ppo_launch] 正在清理…"
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "[train_ppo_launch] 结束 Gazebo 进程组 (pgid=${SIM_PID})"
    # 负 PID 表示整个进程组，确保 gzserver/gzclient/launch 子进程一并结束
    kill -TERM -- "-${SIM_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 环境加载顺序（重要）：
# 1) 先 source ROS + 工作空间 —— 保证 ros2 / 消息包可用
# 2) 再激活 .venv —— 使用其中的 torch 等，同时仍保留已 source 的 ROS 环境变量
# ---------------------------------------------------------------------------

if [[ "${MODE}" == "gazebo" && "${SOURCE_ROS}" == "1" ]]; then
  if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    sourcenoset "/opt/ros/${ROS_DISTRO}/setup.bash"
    echo "[train_ppo_launch] 已 source /opt/ros/${ROS_DISTRO}/setup.bash"
  else
    echo "[train_ppo_launch] 错误: 未找到 /opt/ros/${ROS_DISTRO}/setup.bash（Gazebo 模式必需）"
    exit 1
  fi
  WS_SETUP="${ROOT}/${ROS_WS_INSTALL_REL}/setup.bash"
  if [[ -f "${WS_SETUP}" ]]; then
    sourcenoset "${WS_SETUP}"
    echo "[train_ppo_launch] 已 source ${WS_SETUP}"
  else
    echo "[train_ppo_launch] 错误: 未找到 ${WS_SETUP}，请先: cd ros2_ws && colcon build"
    exit 1
  fi
fi

if [[ "${USE_PROJECT_VENV}" == "1" && -f "${ROOT}/env_obstacle/bin/activate" ]]; then
  sourcenoset "${ROOT}/env_obstacle/bin/activate"
  echo "[train_ppo_launch] 已激活虚拟环境: ${ROOT}/env_obstacle"
fi

# ---------------------------------------------------------------------------
# 启动带 GUI 的 Gazebo（可选）
# ---------------------------------------------------------------------------

if [[ "${MODE}" == "gazebo" && "${START_GAZEBO}" == "1" ]]; then
  echo "[train_ppo_launch] 启动仿真（GUI=${GAZEBO_GUI} world=${WORLD}）…"
  echo "[train_ppo_launch] 日志: ${SIM_LOG}"
  rm -f "${SIM_LOG}"
  # 在 workspace 根下 launch（与 run_launch_and_collect_obs.sh 一致）
  # setsid: 让 launch 成为新进程组组长，退出时可 kill 整个组（含 gzserver/gzclient）
  LAUNCH_FILE=\"${ROOT}/ros2_ws/src/rl_car_gazebo/launch/sim.launch.py\"
  setsid bash -lc "
    cd \"${ROOT}/${ROS_WS_INSTALL_REL}/..\" || cd \"${ROOT}/ros2_ws\"
    set +u
    source \"/opt/ros/${ROS_DISTRO}/setup.bash\"
    source install/setup.bash
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
  echo "[train_ppo_launch] sim.launch.py pid=${SIM_PID}"

  echo "[train_ppo_launch] 等待话题 /scan、/odom、/depth_cam/camera/image_raw …"
  TLIST() { ros2 topic list -t 2>/dev/null || true; }
  _ok=0
  for _ in $(seq 1 240); do
    if TLIST | grep -qE '^/scan ' \
      && TLIST | grep -qE '^/odom ' \
      && TLIST | grep -qE '^/depth_cam/camera/image_raw '; then
      _ok=1
      break
    fi
    sleep 0.5
  done
  if [[ "${_ok}" != "1" ]]; then
    echo "[train_ppo_launch] 错误: 等待传感器话题超时。请查看: ${SIM_LOG}"
    exit 1
  fi

  # 必须确认小车已成功 spawn；否则后续训练不会有 /cmd_vel 反馈
  echo "[train_ppo_launch] 等待 spawn_entity 成功日志…"
  _spawn_ok=0
  for _ in $(seq 1 120); do
    if grep -qE "Successfully spawned entity \[rl_car\]|Spawn success" "${SIM_LOG}" 2>/dev/null; then
      _spawn_ok=1
      break
    fi
    if grep -qE "Service /spawn_entity unavailable|Spawn service failed|process has died" "${SIM_LOG}" 2>/dev/null; then
      echo "[train_ppo_launch] 错误: spawn_entity 失败（/spawn_entity 不可用）"
      echo "[train_ppo_launch] 请查看日志尾部:"
      tail -n 80 "${SIM_LOG}" || true
      exit 1
    fi
    sleep 0.5
  done
  if [[ "${_spawn_ok}" != "1" ]]; then
    echo "[train_ppo_launch] 错误: 超时仍未看到 spawned entity 日志"
    tail -n 80 "${SIM_LOG}" || true
    exit 1
  fi
  echo "[train_ppo_launch] spawn_entity 成功"
  # 按你当前环境：至少等待 /get_model_list；并尽量等待 /gazebo/set_model_state（或 /set_model_state）。
  echo "[train_ppo_launch] 等待 Gazebo 服务（${GAZEBO_GET_MODEL_LIST_SVC}）…"
  _model_list_ok=0
  for _ in $(seq 1 40); do
    sl=$(ros2 service list 2>/dev/null || true)
    if echo "$sl" | grep -qE "^${GAZEBO_GET_MODEL_LIST_SVC}$"; then
      _model_list_ok=1
      break
    fi
    sleep 0.5
  done
  if [[ "${_model_list_ok}" != "1" ]]; then
    echo "[train_ppo_launch] 错误: 未检测到 ${GAZEBO_GET_MODEL_LIST_SVC}，Gazebo 可能未正常启动。"
    exit 1
  fi

  echo "[train_ppo_launch] 检查重置服务（set_model_state / set_entity_state / reset_world）…"
  _set_state_ok=0
  for _ in $(seq 1 20); do
    sl=$(ros2 service list 2>/dev/null || true)
    if echo "$sl" | grep -qE "^${GAZEBO_SET_MODEL_STATE_SVC}$|^/set_model_state$|^/gazebo/set_entity_state$|^/set_entity_state$|^/reset_world$"; then
      _set_state_ok=1
      break
    fi
    sleep 0.5
  done
  if [[ "${_set_state_ok}" == "1" ]]; then
    echo "[train_ppo_launch] 已检测到重置服务（model/entity state 或 reset_world）"
  else
    echo "[train_ppo_launch] 警告: 未检测到任何重置服务；训练可继续，但 reset 能力受限。"
  fi
  sleep "${GAZEBO_SERVICE_SETTLE_SEC}"
  echo "[train_ppo_launch] 即将开始训练（请查看 Gazebo 窗口中小车）"
fi

# ---------------------------------------------------------------------------
# 训练（不使用 exec，以便 EXIT trap 能执行并关闭 Gazebo）
# ---------------------------------------------------------------------------

ENV_FLAG=()
if [[ "${MODE}" == "mock" ]]; then
  ENV_FLAG+=(--mock)
elif [[ "${MODE}" == "gazebo" ]]; then
  ENV_FLAG+=(--gazebo)
else
  echo "[train_ppo_launch] 错误: MODE 必须是 mock 或 gazebo"
  exit 1
fi

if [[ "${SAVE_PATH}" != /* ]]; then
  SAVE_PATH="${ROOT}/${SAVE_PATH}"
fi

echo "[train_ppo_launch] ROOT=${ROOT}"
echo "[train_ppo_launch] MODE=${MODE} START_GAZEBO=${START_GAZEBO} GUI=${GAZEBO_GUI}"
echo "[train_ppo_launch] TOTAL_UPDATES=${TOTAL_UPDATES} ROLLOUT_STEPS=${ROLLOUT_STEPS} DEVICE=${DEVICE}"

set +e
python3 -m rl_algorithms.train_ppo \
  "${ENV_FLAG[@]}" \
  --seed "${SEED}" \
  --total-updates "${TOTAL_UPDATES}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --device "${DEVICE}" \
  --save "${SAVE_PATH}" \
  --gamma "${GAMMA}" \
  --gae-lambda "${GAE_LAMBDA}" \
  --clip-coef "${CLIP_COEF}" \
  --value-coef "${VALUE_COEF}" \
  --entropy-coef "${ENTROPY_COEF}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --lr "${LR}" \
  --ppo-epochs "${PPO_EPOCHS}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
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
  --step-log-csv "${STEP_LOG_CSV}"
RC=$?
set -e

if [[ "${RC}" != "0" ]]; then
  echo "[train_ppo_launch] 训练进程退出码: ${RC}"
fi
exit "${RC}"
