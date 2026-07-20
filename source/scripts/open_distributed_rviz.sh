#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
rviz_config="${project_root}/rviz/distributed_grasp_pipeline.rviz"

if [[ ! -f "${rviz_config}" ]]; then
  echo "RViz config not found: ${rviz_config}" >&2
  exit 1
fi

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
source "${PIPER_ROS_ROOT:-${bundle_root}/piper_ros_ws}/install/setup.bash"
if [[ "${restore_nounset}" -eq 1 ]]; then
  set -u
fi

exec rviz2 -d "${rviz_config}" "$@"
