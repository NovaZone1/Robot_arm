#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
piper_root="${PIPER_ROS_ROOT:-${bundle_root}/piper_ros_ws}"
piper_driver_executable="${piper_root}/install/piper/lib/piper/piper_single_ctrl"

can_port="${PIPER_CAN_PORT:-can0}"
auto_enable="${PIPER_AUTO_ENABLE:-false}"
gripper_exist="${PIPER_GRIPPER_EXIST:-true}"
# The grasp client publishes the physical full-stroke opening in metres
# (70 mm -> 0.070). RViz's paired finger joints use half-stroke semantics, but
# that conversion is handled by the feedback relay and must not be applied to
# /joint_ctrl_single commands a second time.
gripper_val_mutiple="${PIPER_GRIPPER_VAL_MUTIPLE:-1}"
piper_sdk_root="${PIPER_SDK_ROOT:-${bundle_root}/piper_sdk}"
python_can_site="${PIPER_PYTHON_CAN_SITE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --can-port)
      can_port="$2"
      shift 2
      ;;
    --auto-enable)
      auto_enable="$2"
      shift 2
      ;;
    --gripper-exist)
      gripper_exist="$2"
      shift 2
      ;;
    --gripper-val-mutiple)
      gripper_val_mutiple="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: run_piper_driver.sh [options]

Options:
  --can-port <name>             CAN interface to use. Default: ${can_port}
  --auto-enable <true|false>    Whether to auto-enable the arm. Default: ${auto_enable}
  --gripper-exist <true|false>  Whether the gripper is present. Default: ${gripper_exist}
  --gripper-val-mutiple <int>   Gripper scaling. Default: ${gripper_val_mutiple}

Environment overrides:
  PIPER_SDK_ROOT=${piper_sdk_root}
  PIPER_PYTHON_CAN_SITE=<site-packages containing python-can>
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${piper_sdk_root}" ]]; then
  echo "piper_sdk root not found: ${piper_sdk_root}" >&2
  exit 1
fi

# Find piper_single_ctrl in the Piper ROS workspace.
if [[ ! -x "${piper_driver_executable}" ]]; then
  piper_driver_executable="$(find "${piper_root}/install" -name "piper_single_ctrl" -type f 2>/dev/null | head -1 || true)"
fi
if [[ -z "${piper_driver_executable}" || ! -x "${piper_driver_executable}" ]]; then
  echo "piper driver executable not found in ${piper_root}/install" >&2
  echo "Make sure piper_ros is built in ${piper_root}" >&2
  exit 1
fi

if [[ -z "${python_can_site}" ]]; then
  for candidate in \
    "${bundle_root}/.venv/lib/python3.10/site-packages" \
    /usr/local/lib/python3.10/dist-packages \
    /usr/lib/python3/dist-packages
  do
    if [[ -f "${candidate}/can/__init__.py" ]]; then
      python_can_site="${candidate}"
      break
    fi
  done
fi

if [[ ! -f "${python_can_site}/can/__init__.py" ]]; then
  echo "python-can package not found. Set PIPER_PYTHON_CAN_SITE to a site-packages dir containing can/__init__.py" >&2
  exit 1
fi

if ! ip link show "${can_port}" >/dev/null 2>&1; then
  echo "CAN interface ${can_port} was not found." >&2
  echo "Connect the USB-CAN adapter or pass --can-port with the actual interface name." >&2
  exit 1
fi
if ! ip link show "${can_port}" | grep -q "UP"; then
  echo "CAN interface ${can_port} exists but is not UP." >&2
  echo "Bring it up before starting the Piper driver:" >&2
  echo "  sudo ip link set ${can_port} down" >&2
  echo "  sudo ip link set ${can_port} type can bitrate 1000000" >&2
  echo "  sudo ip link set ${can_port} up" >&2
  exit 1
fi

export ROS_HOME="${ROS_HOME:-${bundle_root}/tmp/piper_ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${ROS_HOME}/log}"
mkdir -p "${ROS_LOG_DIR}"

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
source "${piper_root}/install/setup.bash"
if [[ "${restore_nounset}" -eq 1 ]]; then
  set -u
fi
export PYTHONPATH="${piper_sdk_root}:${python_can_site}:${PYTHONPATH:-}"

echo "Starting AgileX Piper driver"
echo "  can_port: ${can_port}"
echo "  auto_enable: ${auto_enable}"
echo "  gripper_exist: ${gripper_exist}"
echo "  gripper_val_mutiple: ${gripper_val_mutiple}"
echo "  piper_sdk_root: ${piper_sdk_root}"
echo "  python_can_site: ${python_can_site}"
echo "  driver: ${piper_driver_executable}"

exec "${piper_driver_executable}" --ros-args \
  -p "can_port:=${can_port}" \
  -p "auto_enable:=${auto_enable}" \
  -p "gripper_exist:=${gripper_exist}" \
  -p "gripper_val_mutiple:=${gripper_val_mutiple}"
