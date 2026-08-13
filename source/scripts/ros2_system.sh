#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
workspace_root="${ROBOT_GRASP_WORKSPACE_ROOT:-${bundle_root}/ros_ws}"
piper_workspace_root="${ROBOT_GRASP_PIPER_WORKSPACE_ROOT:-${bundle_root}/piper_ros_ws}"

# Keep ros2 CLI on the system Python side; py310 overlays are only for runtime nodes.
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
if [[ -f "${piper_workspace_root}/install/setup.bash" ]]; then
  source "${piper_workspace_root}/install/setup.bash"
fi
source "${workspace_root}/install/setup.bash"
if [ "${restore_nounset}" -eq 1 ]; then
  set -u
fi

exec ros2 "$@"
