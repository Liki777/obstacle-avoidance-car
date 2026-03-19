#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/graduation_project"
WS="${ROOT}/ros2_ws"

WORLD="${1:-maze.world}"
GUI="${2:-true}"           # true/false
X="${3:-0.0}"
Y="${4:-0.0}"
Z="${5:-0.1}"

SIM_LOG="/tmp/rl_car_sim.log"

cleanup() {
  echo ""
  echo "[run_sim] cleaning up..."
  jobs -pr | xargs -r kill || true
  wait || true
}
trap cleanup EXIT INT TERM

cd "${WS}"

# ROS setup scripts are not always nounset-safe.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

rm -f "${SIM_LOG}"

echo "[run_sim] launching gazebo + spawn"
echo "[run_sim]   world=${WORLD} gui=${GUI} xyz=(${X}, ${Y}, ${Z})"
ros2 launch rl_car_gazebo sim.launch.py \
  "world:=${WORLD}" \
  "gui:=${GUI}" \
  "x:=${X}" \
  "y:=${Y}" \
  "z:=${Z}" \
  >"${SIM_LOG}" 2>&1 &
LAUNCH_PID=$!

echo "[run_sim] launch pid=${LAUNCH_PID} (logs: ${SIM_LOG})"
echo "[run_sim] waiting for /scan + /odom topics..."
for _ in {1..120}; do
  if ros2 topic list -t 2>/dev/null | rg -q '^/scan ' \
    && ros2 topic list -t 2>/dev/null | rg -q '^/odom '; then
    break
  fi
  sleep 0.5
done

echo "[run_sim] waiting for spawn success..."
for _ in {1..120}; do
  if rg -q "Spawn success|Successfully spawned entity \\[rl_car\\]" "${SIM_LOG}" 2>/dev/null; then
    break
  fi
  if rg -q "Traceback|ModuleNotFoundError|process has died|Spawn failed|Service /spawn_entity not available" "${SIM_LOG}" 2>/dev/null; then
    echo "[run_sim] launch error detected. Last 120 log lines:"
    tail -n 120 "${SIM_LOG}" || true
    exit 1
  fi
  sleep 0.5
done

echo "[run_sim] starting smoke test (cmd_vel publish + sensors monitor)"
/usr/bin/python3 "${ROOT}/scripts/smoketest.py" &
SMOKE_PID=$!

echo "[run_sim] smoke pid=${SMOKE_PID}"
echo ""
echo "[run_sim] tips:"
echo "  - Ctrl+C to stop everything"
echo "  - tail -f ${SIM_LOG}"
echo "  - ros2 topic list -t | rg 'scan|odom|depth_cam|cmd_vel'"
echo ""

wait

