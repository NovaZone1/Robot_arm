#!/usr/bin/env bash
# Record placement mapping samples: label (u, v) at the fixed observation
# pose, then the taught release TCP after the operator jogs the arm.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
# ROS nodes and this recorder must use system Python (rclpy).
# ros_env_graspnet.sh prepends the conda/venv bin onto PATH.
export ROBOT_GRASP_SYSTEM_PYTHON="${ROBOT_GRASP_SYSTEM_PYTHON:-/usr/bin/python3}"
# shellcheck disable=SC1091
source "${script_dir}/ros_env_graspnet.sh"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
cd "${project_root}"
exec /usr/bin/python3 "${script_dir}/record_placement_uv_xy.py" "$@"
