#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <text-prompt> [extra run_grasp_pipeline_ros2.py args...]" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

source "${script_dir}/ros_env_graspnet.sh"

exec python "${project_root}/src/run_grasp_pipeline_ros2.py" \
  "$@" \
  --robot-backend fake
