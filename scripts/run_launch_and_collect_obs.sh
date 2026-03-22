#!/usr/bin/env bash
# 一键：启动 Gazebo+小车 + 控制(smoketest) + 目标发布(goal_pose) + 采集(obs 全来自话题订阅)
set -euo pipefail

ROOT="${HOME}/graduation_project"
WS="${ROOT}/ros2_ws"

WORLD="${1:-maze.world}"
DURATION="${2:-60}"
SAMPLE_HZ="${3:-10}"
GOAL_X="${4:-2.0}"
GOAL_Y="${5:-2.0}"
OUTPUT="${6:-${ROOT}/logs/rl_car_obs.csv}"

GUI="${GUI:-true}"
GOAL_FRAME="${GOAL_FRAME:-odom}"

SIM_LOG="${ROOT}/logs/rl_car_sim.log"

cleanup() {
  echo ""
  echo "[run_obs] cleaning up..."
  jobs -pr | xargs -r kill || true
  wait || true
}
trap cleanup EXIT INT TERM

mkdir -p "${ROOT}/logs"
cd "${WS}"

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

rm -f "${SIM_LOG}"
rm -f "${OUTPUT}"

echo "[run_obs] launching gazebo+spawn (gui=${GUI})..."
ros2 launch rl_car_gazebo sim.launch.py \
  "world:=${WORLD}" \
  "gui:=${GUI}" \
  >"${SIM_LOG}" 2>&1 &
echo "[run_obs] sim log: ${SIM_LOG}"

echo "[run_obs] waiting for /scan /odom /depth_cam/camera/image_raw ..."
TLIST() { ros2 topic list -t 2>/dev/null || true; }
for _ in {1..240}; do
  if TLIST | grep -qE '^/scan ' \
    && TLIST | grep -qE '^/odom ' \
    && TLIST | grep -qE '^/depth_cam/camera/image_raw '; then
    break
  fi
  sleep 0.5
done

echo "[run_obs] starting goal publisher -> /goal_pose (x=${GOAL_X} y=${GOAL_Y} frame=${GOAL_FRAME})"
/usr/bin/python3 "${ROOT}/scripts/goal_pose_publisher.py" \
  --x "${GOAL_X}" --y "${GOAL_Y}" --frame-id "${GOAL_FRAME}" --rate 5 &
GOAL_PUB_PID=$!

sleep 0.5
echo "[run_obs] starting smoketest (cmd_vel)..."
/usr/bin/python3 "${ROOT}/scripts/smoketest.py" &

echo "[run_obs] starting observation collector (subscribe only, no random)..."
/usr/bin/python3 "${ROOT}/scripts/collect_observations.py" \
  --duration "${DURATION}" \
  --sample-hz "${SAMPLE_HZ}" \
  --scan-topic /scan \
  --odom-topic /odom \
  --camera-topic /depth_cam/camera/image_raw \
  --goal-topic /goal_pose \
  --lidar-dim 15 \
  --lidar-reduce min \
  --output "${OUTPUT}" &
COLLECT_PID=$!

echo "[run_obs] collector pid=${COLLECT_PID}"
wait "${COLLECT_PID}" || true

echo "[run_obs] done. tail csv:"
tail -n 15 "${OUTPUT}" 2>/dev/null || true
