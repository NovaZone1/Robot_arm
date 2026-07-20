#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
runner_python="${ROBOT_GRASP_SYSTEM_PYTHON:-/usr/bin/python3}"

cd "${project_root}"
exec "${runner_python}" -m robot_grasp_ros2.live_grasp_one_click --once "$@"
