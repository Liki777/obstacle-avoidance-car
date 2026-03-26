#!/usr/bin/env bash
# =============================================================================
# 一键验证：启动 Gazebo GUI + 小车 + 持续前进 + 检查传感器
#
# - 不使用任何 .pt
# - 用 Gazebo GUI 可视化车头方向
# - 终端周期性打印雷达/深度信息
#
# 用法：
#   chmod +x scripts/run_drive_forward_check.sh
#   ./scripts/run_drive_forward_check.sh
#
# 常用改动（直接改下方变量）：
#   WORLD=empty.world 或 maze.world
#   LINEAR_X 正数/负数（看哪边是“车头”）
#   ANGULAR_Z 给一个小值让车转圈，方便看朝向与深度变化
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROOT}/ros2_ws"
ROS_DISTRO="${ROS_DISTRO:-humble}"

WORLD="${WORLD:-empty.world}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"

SPAWN_X="${SPAWN_X:-0.0}"
SPAWN_Y="${SPAWN_Y:-0.0}"
SPAWN_YAW="${SPAWN_YAW:-0.0}"

LINEAR_X="${LINEAR_X:-0.4}"
ANGULAR_Z="${ANGULAR_Z:-0.0}"

SCAN_TOPIC="${SCAN_TOPIC:-/scan}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/depth_cam/camera/depth/image_raw}"
CMD_VEL_TOPIC="${CMD_VEL_TOPIC:-/cmd_vel}"

SIM_LOG="${ROOT}/logs/drive_check_sim.log"

SIM_PID=""
cleanup() {
  echo ""
  echo "[drive_check] cleaning up..."
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill -TERM -- "-${SIM_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "${ROOT}/logs"

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${WS}/install/setup.bash"
set -u

# 若你用的是项目里的 conda/venv 环境，这里自动激活（可删）
if [[ -f "${ROOT}/env_obstacle/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env_obstacle/bin/activate"
fi

echo "[drive_check] launching gazebo gui=${GAZEBO_GUI} world=${WORLD}"
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
echo "[drive_check] sim pid=${SIM_PID} log=${SIM_LOG}"

echo "[drive_check] waiting for topics: ${SCAN_TOPIC} /odom ${DEPTH_TOPIC} ..."
TLIST() { ros2 topic list -t 2>/dev/null || true; }
for _ in $(seq 1 240); do
  if TLIST | grep -qE "^${SCAN_TOPIC} " && TLIST | grep -qE '^/odom ' && TLIST | grep -qE "^${DEPTH_TOPIC} "; then
    break
  fi
  sleep 0.5
done

echo "[drive_check] driving now. Try LINEAR_X=0.4 then -0.4 to confirm front."
cd "${ROOT}"
python3 "${ROOT}/scripts/drive_forward_and_check_sensors.py" \
  --cmd-vel-topic "${CMD_VEL_TOPIC}" \
  --scan-topic "${SCAN_TOPIC}" \
  --depth-topic "${DEPTH_TOPIC}" \
  --linear-x "${LINEAR_X}" \
  --angular-z "${ANGULAR_Z}" \
  --print-every-sec 0.5

