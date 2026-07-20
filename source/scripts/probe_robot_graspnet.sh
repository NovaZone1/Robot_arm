#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

source "${script_dir}/ros_env_graspnet.sh"

exec python "${project_root}/src/run_grasp_pipeline_ros2.py" \
  --robot-backend ros2 \
  --probe-robot \
  --can "${PIPER_CAN_PORT:-can0}" \
  "$@"
