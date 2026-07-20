#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
piper_root="${PIPER_ROS_ROOT:-${bundle_root}/piper_ros_ws}"
project_ws="${ROBOT_GRASP_WORKSPACE_ROOT:-${bundle_root}/ros_ws}"

if [[ ! -d "${piper_root}" ]]; then
  echo "Piper ROS workspace not found: ${piper_root}" >&2
  exit 1
fi

export LC_NUMERIC="${LC_NUMERIC:-en_US.UTF-8}"

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
if [[ -f "${piper_root}/install/setup.bash" ]]; then
  source "${piper_root}/install/setup.bash"
fi
# Project overlay if available.
if [[ -f "${project_ws}/install/setup.bash" ]]; then
  source "${project_ws}/install/setup.bash"
elif [[ -f "${bundle_root}/install/setup.bash" ]]; then
  source "${bundle_root}/install/setup.bash"
fi
if [[ "${restore_nounset}" -eq 1 ]]; then
  set -u
fi

if ! ros2 pkg prefix piper_description >/dev/null 2>&1; then
  echo "piper_description is not in the current overlay. Building it first..."
  (
    cd "${piper_root}"
    colcon build \
      --merge-install \
      --packages-select piper_description \
      --cmake-clean-cache \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
  )
  restore_nounset=0
  if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
  fi
  source "${piper_root}/install/setup.bash"
  if [[ -f "${project_ws}/install/setup.bash" ]]; then
    source "${project_ws}/install/setup.bash"
  elif [[ -f "${bundle_root}/install/setup.bash" ]]; then
    source "${bundle_root}/install/setup.bash"
  fi
  if [[ "${restore_nounset}" -eq 1 ]]; then
    set -u
  fi
fi

if ! ros2 pkg prefix robot_grasp_ros2 >/dev/null 2>&1; then
  echo "robot_grasp_ros2 is not in the current overlay." >&2
  echo "Expected project overlay: ${project_ws}/install/setup.bash" >&2
  echo "Build/source the grasp workspace first, for example:" >&2
  echo "  cd ${project_ws}" >&2
  echo "  colcon build --symlink-install --packages-select robot_grasp_msgs robot_grasp_ros2" >&2
  exit 1
fi

echo "Starting MoveIt IK service via robot_grasp_ros2/piper_moveit_ik.launch.py"
echo "  piper_root: ${piper_root}"
echo "  project_ws: ${project_ws}"
echo "  LC_NUMERIC: ${LC_NUMERIC}"

exec ros2 launch robot_grasp_ros2 piper_moveit_ik.launch.py "$@"
