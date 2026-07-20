#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${script_dir}/ros_env_graspnet.sh"

cd "${ROBOT_GRASP_ROS2_ROOT}"
export PYTHONPATH="${ROBOT_GRASP_ROS2_ROOT}:${PYTHONPATH:-}"

if [[ "${ALLOW_DUPLICATE_GRASP_PIPELINE:-0}" != "1" ]]; then
  existing_processes="$(pgrep -af "python -m robot_grasp_ros2.grasp_pipeline_node" || true)"
  if [[ -n "${existing_processes}" ]]; then
    echo "Another grasp_pipeline node is already running. Stop it before starting a new one." >&2
    echo "${existing_processes}" >&2
    echo "If you really need a duplicate node, rerun with ALLOW_DUPLICATE_GRASP_PIPELINE=1." >&2
    exit 1
  fi
fi

exec python -m robot_grasp_ros2.grasp_pipeline_node "$@"
