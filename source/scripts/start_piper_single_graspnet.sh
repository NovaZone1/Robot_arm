#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${script_dir}/ros_env_graspnet.sh"

can_port="${PIPER_CAN_PORT:-can0}"
auto_enable="${PIPER_AUTO_ENABLE:-false}"
gripper_exist="${PIPER_GRIPPER_EXIST:-true}"
gripper_multiple="${PIPER_GRIPPER_MULTIPLE:-2}"
log_level="${PIPER_LOG_LEVEL:-warn}"

if ! ip link show "${can_port}" >/dev/null 2>&1; then
  echo "CAN interface not found: ${can_port}" >&2
  echo "Available CAN interfaces:" >&2
  ip -details link show type can >&2 || true
  echo "Bring up the adapter first, or export PIPER_CAN_PORT with the real interface name." >&2
  exit 1
fi

echo "starting piper_single_ctrl"
echo "  can_port: ${can_port}"
echo "  auto_enable: ${auto_enable}"
echo "  gripper_exist: ${gripper_exist}"
echo "  gripper_val_mutiple: ${gripper_multiple}"
echo "  log_level: ${log_level}"

exec ros2 run piper piper_single_ctrl --ros-args \
  -p "can_port:=${can_port}" \
  -p "auto_enable:=${auto_enable}" \
  -p "gripper_exist:=${gripper_exist}" \
  -p "gripper_val_mutiple:=${gripper_multiple}" \
  --log-level "${log_level}" \
  "$@"
