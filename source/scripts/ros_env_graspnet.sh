#!/usr/bin/env bash
set -euo pipefail

_robot_grasp_ros2_setup_env() {
  local script_dir project_root bundle_root workspace_root outer_workspace_root install_root env_root piper_overlay project_overlay
  local restore_nounset
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_root="$(cd "${script_dir}/.." && pwd)"
  bundle_root="$(cd "${project_root}/.." && pwd)"
  workspace_root="${ROBOT_GRASP_WORKSPACE_ROOT:-${bundle_root}/ros_ws}"
  # Use the colcon workspace as the project overlay root
  _colcon_ws="${workspace_root}"
  env_root="${ROBOT_GRASP_PYTHON_ENV_ROOT:-${bundle_root}/.venv}"
  piper_overlay="${PIPER_ROS_ROOT:-${bundle_root}/piper_ros_ws}/install/setup.bash"
  project_overlay="${_colcon_ws}/install/setup.bash"

  if [[ ! -d "${env_root}" ]]; then
    echo "perception Python environment not found: ${env_root}" >&2
    return 1
  fi
  if [[ ! -f "${piper_overlay}" ]]; then
    echo "piper_ws overlay not found: ${piper_overlay}" >&2
    return 1
  fi

  export ROBOT_GRASP_ROS2_ROOT="${project_root}"
  export ROBOT_GRASP_WORKSPACE_ROOT="${_colcon_ws}"
  export ROBOT_GRASP_OUTER_WORKSPACE_ROOT="${_colcon_ws}"
  export ROBOT_GRASP_SYSTEM_PYTHON="/usr/bin/python3"
  export ROBOT_GRASP_CONDA_PYTHON="${env_root}/bin/python"
  export PATH="${env_root}/bin:${PATH}"
  export LD_LIBRARY_PATH="${env_root}/lib:${LD_LIBRARY_PATH:-}"

  restore_nounset=0
  if [[ $- == *u* ]]; then
    restore_nounset=1
    set +u
  fi
  source /opt/ros/humble/setup.bash
  source "${piper_overlay}"
  if [[ -f "${project_overlay}" ]]; then
    source "${project_overlay}"
  fi
  if [[ "${restore_nounset}" -eq 1 ]]; then
    set -u
  fi

  export PATH="${env_root}/bin:${PATH}"
  export PYTHONPATH="${project_root}:${PYTHONPATH:-}"

  # GraspNet checkpoints — set only if they exist on this machine.
  # YOLOv8-seg weights are resolved by Ultralytics.
  _graspnet_baseline="${GRASPNET_BASELINE_ROOT:-${bundle_root}/graspnet}"
  _graspnet_ckpt="${_graspnet_baseline}/checkpoint-rs.tar"

  export GRASPNET_BASELINE_ROOT="${_graspnet_baseline}"
  if [[ -f "${_graspnet_ckpt}" ]]; then
    export GRASPNET_CHECKPOINT="${_graspnet_ckpt}"
  fi

  echo "robot_grasp_ros2 ROS env ready"
  echo "  python: $(command -v python)"
  echo "  ros python: ${ROBOT_GRASP_SYSTEM_PYTHON}"
  echo "  worker python: ${ROBOT_GRASP_CONDA_PYTHON}"
  echo "  ros2: $(command -v ros2)"
  echo "  project: ${project_root}"
  echo "  workspace: ${workspace_root}"
  echo "  piper install: ${piper_overlay}"
  if [[ -f "${project_overlay}" ]]; then
    echo "  project install: ${_colcon_ws}/install"
  else
    echo "  project install: ${_colcon_ws}/install (missing; run colcon build in ${_colcon_ws})"
  fi
  echo "  segmentation: YOLOv8-seg (auto-download)"
  if [[ -n "${GRASPNET_CHECKPOINT:-}" ]]; then
    echo "  graspnet checkpoint: ${GRASPNET_CHECKPOINT}"
  else
    echo "  graspnet checkpoint: not found (GraspNet disabled until installed)"
  fi
  echo "  note: use ${project_root}/scripts/ros2_system.sh for ros2 CLI commands"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  _robot_grasp_ros2_setup_env
  return 0
fi

echo "Use this file with: source <piper_grasp_project>/source/scripts/ros_env_graspnet.sh" >&2
exit 1
