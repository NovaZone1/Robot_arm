#!/usr/bin/env bash
# Block until a waved red flag is verified. This script never starts Nav2 or
# the grasp pipeline itself. Observation-pose replay stays disabled until a
# newly taught pose has been verified from live Piper feedback.
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
params_file="${project_root}/config/distributed/red_flag_start.params.yaml"
ros2_cli="${ROBOT_GRASP_ROS2_CLI:-${script_dir}/ros2_system.sh}"

# ROS setup files are not nounset-safe: Humble's setup.bash reads optional
# AMENT_* variables before defining them.  Load the environments first, then
# enable strict unset-variable checking for this script's own logic.
source /opt/ros/humble/setup.bash
source "${bundle_root}/piper_ros_ws/install/setup.bash"
source "${bundle_root}/ros_ws/install/setup.bash"
set -u

if [[ "${RED_FLAG_MOVE_TO_OBSERVATION:-1}" == "1" ]]; then
  echo "moving arm to verified red-flag observation pose"
  move_attempts="${RED_FLAG_MOVE_ATTEMPTS:-2}"
  if ! [[ "${move_attempts}" =~ ^[1-2]$ ]]; then
    echo "red-flag gate blocked: RED_FLAG_MOVE_ATTEMPTS must be 1 or 2" >&2
    exit 22
  fi
  move_output=""
  move_succeeded=false
  for ((attempt = 1; attempt <= move_attempts; attempt++)); do
    move_output="$(${ros2_cli} service call \
      /robot_executor/execute_named_pose \
      robot_grasp_msgs/srv/ExecuteNamedPose \
      "{name: 'red_flag_observation', pose: {x_mm: 44.300, y_mm: -3.710, z_mm: 596.805, roll_deg: 3.037, pitch_deg: 72.962, yaw_deg: 0.898}, speed_percent: 25.0, open_gripper_first: false}" \
      2>&1)"
    if grep -Eqi 'success[=:][[:space:]]*true' <<<"${move_output}"; then
      move_succeeded=true
      break
    fi
    if ((attempt < move_attempts)); then
      echo "red-flag observation command did not start; confirming enable and retrying once" >&2
      enable_output="$(${ros2_cli} service call \
        /enable_srv piper_msgs/srv/Enable \
        "{enable_request: true}" 2>&1 || true)"
      if ! grep -Eqi 'enable_response[=:][[:space:]]*true' <<<"${enable_output}"; then
        echo "red-flag gate blocked: Piper re-enable failed" >&2
        echo "${enable_output}" >&2
        exit 23
      fi
      sleep 1
    fi
  done
  if [[ "${move_succeeded}" != true ]]; then
    echo "red-flag gate blocked: arm could not reach observation pose" >&2
    echo "${move_output}" >&2
    exit 20
  fi
elif [[ "${RED_FLAG_MOVE_TO_OBSERVATION}" != "0" ]]; then
  echo "red-flag gate blocked: RED_FLAG_MOVE_TO_OBSERVATION must be 0 or 1" >&2
  exit 21
else
  echo "red-flag observation replay disabled by RED_FLAG_MOVE_TO_OBSERVATION=0"
fi

echo "red-flag gate armed; wave the flag until detection is confirmed"
ros2 run robot_grasp_ros2 red_flag_start_gate \
  --ros-args --params-file "${params_file}"

# Do not hold navigation while the arm folds away.  Once the start signal has
# been verified, dispatch Home asynchronously and return success immediately;
# the route caller can start driving in parallel.
home_log="${bundle_root}/ros_ws/log/red_flag_home_$(date +%Y%m%d_%H%M%S).log"
nohup "${ros2_cli}" service call \
  /robot_executor/execute_named_pose \
  robot_grasp_msgs/srv/ExecuteNamedPose \
  "{name: 'home_after_red_flag', pose: {x_mm: 57.0, y_mm: 0.0, z_mm: 215.0, roll_deg: 0.0, pitch_deg: 85.0, yaw_deg: 0.0}, speed_percent: 25.0, open_gripper_first: false}" \
  >"${home_log}" 2>&1 &
home_pid=$!
echo "red flag confirmed; Home dispatched in parallel with navigation pid=${home_pid} log=${home_log}"
