#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_distributed_stack_graspnet.sh [options]

Options:
  --robot-backend <fake|ros2>  Override robot_executor backend.
  --pose-execution-mode <direct|moveit_ik>
                               Override robot_executor pose execution mode.
  --prompt <text>              Set orchestrator prompt and auto-start once.
  --execute                    Execute the final grasp plan instead of dry-run.
  --confirm                    Wait for /grasp_pipeline/confirm before executing the final plan.
  --precenter                  Enable distributed precenter before final planning.
  --enable-pregrasp            Enable the pregrasp waypoint in both planner and executor.
  --show-pointcloud            Ask vision_worker to pop the Open3D preview window during analysis.
  --with-piper-driver          Also start piper_single_ctrl driver in background (ros2 backend only).
  --with-moveit-ik             Also start MoveIt IK service in background (moveit_ik mode only).
  --warmup                     Warm up camera and vision daemons after nodes are ready.
  --log-root <dir>             Override log root directory.
  --help                       Show this help.

Examples:
  # fake backend (no hardware needed)
  ./scripts/run_distributed_stack_graspnet.sh --robot-backend fake --prompt cup

  # ros2 backend, manual driver startup
  ./scripts/run_piper_driver.sh          # terminal A
  ./scripts/run_piper_moveit_ik.sh       # terminal B
  ./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik --execute --prompt cup

  # ros2 backend, all-in-one
  ./scripts/run_distributed_stack_graspnet.sh \
    --robot-backend ros2 --pose-execution-mode moveit_ik \
    --with-piper-driver --with-moveit-ik \
    --execute --prompt cup
EOF
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"
new_workspace_root="${ROBOT_GRASP_WORKSPACE_ROOT:-${bundle_root}/ros_ws}"
default_log_root="${new_workspace_root}/log/distributed"

robot_backend=""
pose_execution_mode=""
prompt=""
execute_flag=0
confirm_flag=0
precenter_flag=0
enable_pregrasp_flag=0
show_pointcloud_flag=0
with_piper_driver=0
with_moveit_ik=0
warmup_flag=0
log_root="${default_log_root}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-backend)
      [[ $# -ge 2 ]] || usage
      robot_backend="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -ge 2 ]] || usage
      prompt="$2"
      shift 2
      ;;
    --pose-execution-mode)
      [[ $# -ge 2 ]] || usage
      pose_execution_mode="$2"
      shift 2
      ;;
    --execute)
      execute_flag=1
      shift
      ;;
    --confirm)
      confirm_flag=1
      shift
      ;;
    --precenter)
      precenter_flag=1
      shift
      ;;
    --enable-pregrasp)
      enable_pregrasp_flag=1
      shift
      ;;
    --show-pointcloud)
      show_pointcloud_flag=1
      shift
      ;;
    --with-piper-driver)
      with_piper_driver=1
      shift
      ;;
    --with-moveit-ik)
      with_moveit_ik=1
      shift
      ;;
    --warmup)
      warmup_flag=1
      shift
      ;;
    --log-root)
      [[ $# -ge 2 ]] || usage
      log_root="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

existing_nodes="$(pgrep -af 'robot_grasp_ros2\.(camera_server_node|vision_worker_node|robot_executor_node|pipeline_orchestrator_node|scout_scan_controller_node)' || true)"
if [[ -n "${existing_nodes}" ]]; then
  cat >&2 <<EOF
Existing distributed robot_grasp_ros2 nodes are already running.
Stop them before starting a new stack, otherwise services/topics can be mixed across old and new processes.

Detected processes:
${existing_nodes}
EOF
  exit 1
fi

source "${script_dir}/ros_env_graspnet.sh"
ros_python="${ROBOT_GRASP_SYSTEM_PYTHON:-/usr/bin/python3}"

timestamp="$(date +%Y%m%d_%H%M%S)"
session_root="${log_root}/${timestamp}"
mkdir -p "${session_root}"
export ROS_HOME="${session_root}/ros_home"
export ROS_LOG_DIR="${session_root}/ros_home/log"
mkdir -p "${ROS_LOG_DIR}"

camera_params="${project_root}/config/distributed/camera_server.params.yaml"
vision_params="${project_root}/config/distributed/vision_worker.params.yaml"
robot_params="${project_root}/config/distributed/robot_executor.params.yaml"
orchestrator_params="${project_root}/config/distributed/pipeline_orchestrator.params.yaml"
base_scan_params="${project_root}/config/distributed/scout_scan_controller.params.yaml"

pids=()
# Optional dependencies are launched first, then waited for only after the
# core ROS nodes are already available.  Keeping these flags lets their
# readiness checks run concurrently instead of serially delaying all nodes.
wait_for_piper_service=0
wait_for_moveit_service=0

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
}

trap cleanup EXIT INT TERM

start_node() {
  local name="$1"
  shift
  local logfile="${session_root}/${name}.log"
  (
    cd "${project_root}"
    "$@"
  ) >"${logfile}" 2>&1 &
  local pid=$!
  pids+=("${pid}")
  printf '[distributed] started %-24s pid=%s log=%s\n' "${name}" "${pid}" "${logfile}"
}

# ---------------------------------------------------------------------------
# Wait for a ROS2 service to become available (polls every 2s up to timeout).
# Usage: wait_for_service <service_name> <timeout_s> <label>
# ---------------------------------------------------------------------------
wait_for_service() {
  local svc="$1"
  local timeout_s="$2"
  local label="${3:-${svc}}"
  local deadline=$(( $(date +%s) + timeout_s ))
  printf '[distributed] waiting for %-36s' "${label}..."
  while true; do
    if "${project_root}/scripts/ros2_system.sh" service list 2>/dev/null | grep -qF "${svc}"; then
      echo " ready"
      return 0
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
      echo " TIMEOUT after ${timeout_s}s"
      return 1
    fi
    sleep 2
  done
}

# ---------------------------------------------------------------------------
# Step 1: Optionally start Piper driver (ros2 backend only)
# ---------------------------------------------------------------------------
if [[ "${with_piper_driver}" -eq 1 ]]; then
  if [[ "${robot_backend}" != "ros2" ]]; then
    echo "[distributed] --with-piper-driver ignored (robot_backend is not ros2)"
  else
    existing_driver="$(pgrep -af 'piper_single_ctrl' || true)"
    if [[ -n "${existing_driver}" ]]; then
      echo "[distributed] Piper driver already running, skipping start"
    else
      echo "[distributed] Starting Piper driver..."
      piper_logfile="${session_root}/piper_driver.log"
      (
        # run_piper_driver.sh uses exec, so we wrap it in a subshell
        bash "${script_dir}/run_piper_driver.sh"
      ) >"${piper_logfile}" 2>&1 &
      piper_pid=$!
      pids+=("${piper_pid}")
      printf '[distributed] started %-24s pid=%s log=%s\n' "piper_driver" "${piper_pid}" "${piper_logfile}"

      wait_for_piper_service=1
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Step 2: Optionally start MoveIt IK service
# ---------------------------------------------------------------------------
if [[ "${with_moveit_ik}" -eq 1 ]]; then
  if [[ "${pose_execution_mode}" != "moveit_ik" ]]; then
    echo "[distributed] --with-moveit-ik ignored (pose_execution_mode is not moveit_ik)"
  else
    existing_ik="$(pgrep -af 'piper_moveit_ik\|move_group' || true)"
    if [[ -n "${existing_ik}" ]]; then
      echo "[distributed] MoveIt IK service already running, skipping start"
    else
      echo "[distributed] Starting MoveIt IK service..."
      moveit_logfile="${session_root}/moveit_ik.log"
      (
        bash "${script_dir}/run_piper_moveit_ik.sh"
      ) >"${moveit_logfile}" 2>&1 &
      moveit_pid=$!
      pids+=("${moveit_pid}")
      printf '[distributed] started %-24s pid=%s log=%s\n' "moveit_ik" "${moveit_pid}" "${moveit_logfile}"

      wait_for_moveit_service=1
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Step 3: Start the four distributed nodes
# ---------------------------------------------------------------------------
# Keep grasp-time RGB processing consistent with the image set used to train
# the current object/box classifiers.  Those samples were collected with the
# D435's auto white-balance and exposure enabled.  A caller may still opt into
# a separately calibrated fixed profile by explicitly exporting
# D435_LOCK_COLOR=1 (and D435_WHITE_BALANCE/D435_EXPOSURE) before launch.
export D435_LOCK_COLOR="${D435_LOCK_COLOR:-0}"

camera_cmd=(
  "${ros_python}" -m robot_grasp_ros2.camera_server_node
  --ros-args
  --params-file "${camera_params}"
)

vision_cmd=(
  "${ros_python}" -m robot_grasp_ros2.vision_worker_node
  --ros-args
  --params-file "${vision_params}"
)

robot_cmd=(
  "${ros_python}" -m robot_grasp_ros2.robot_executor_node
  --ros-args
  --params-file "${robot_params}"
)
if [[ -n "${robot_backend}" ]]; then
  robot_cmd+=(-p "robot_backend:=${robot_backend}")
fi
if [[ -n "${pose_execution_mode}" ]]; then
  robot_cmd+=(-p "pose_execution_mode:=${pose_execution_mode}")
fi
if [[ "${enable_pregrasp_flag}" -eq 1 ]]; then
  robot_cmd+=(-p "enable_pregrasp:=true")
fi

orchestrator_cmd=(
  "${ros_python}" -m robot_grasp_ros2.pipeline_orchestrator_node
  --ros-args
  --params-file "${orchestrator_params}"
  -p "hand_eye_config:=${project_root}/config/hand_eye/verify_config_eyeinhand_cam2tcp.yaml"
)

base_scan_cmd=(
  "${ros_python}" -m robot_grasp_ros2.scout_scan_controller_node
  --ros-args
  --params-file "${base_scan_params}"
)
if [[ -n "${prompt}" ]]; then
  orchestrator_cmd+=(-p "prompt:=${prompt}" -p "auto_start:=true")
fi
if [[ "${execute_flag}" -eq 1 ]]; then
  orchestrator_cmd+=(-p "execute:=true")
fi
if [[ "${confirm_flag}" -eq 1 ]]; then
  orchestrator_cmd+=(-p "confirm:=true")
fi
if [[ "${precenter_flag}" -eq 1 ]]; then
  orchestrator_cmd+=(-p "precenter:=true")
fi
if [[ "${enable_pregrasp_flag}" -eq 1 ]]; then
  orchestrator_cmd+=(-p "enable_pregrasp:=true")
fi
if [[ "${show_pointcloud_flag}" -eq 1 ]]; then
  orchestrator_cmd+=(-p "show_pointcloud:=true")
fi

start_node "camera_server" "${camera_cmd[@]}"
start_node "vision_worker" "${vision_cmd[@]}"
start_node "robot_executor" "${robot_cmd[@]}"
start_node "base_scan_controller" "${base_scan_cmd[@]}"
start_node "grasp_pipeline" "${orchestrator_cmd[@]}"

# Piper and MoveIt were launched above and have been initializing while the
# camera/vision/executor/pipeline nodes came up.  Check both in parallel here;
# this replaces the former 30s + 60s serial startup delay.
dependency_wait_pids=()
if [[ "${wait_for_piper_service}" -eq 1 ]]; then
  (
    if ! wait_for_service "/enable_srv" 30 "/enable_srv"; then
      echo "[distributed] WARNING: /enable_srv not available after 30s. Check piper_driver.log" >&2
    fi
  ) &
  dependency_wait_pids+=("$!")
fi
if [[ "${wait_for_moveit_service}" -eq 1 ]]; then
  (
    if ! wait_for_service "/compute_ik" 60 "/compute_ik"; then
      echo "[distributed] WARNING: /compute_ik not available after 60s. Check moveit_ik.log" >&2
    fi
  ) &
  dependency_wait_pids+=("$!")
fi
if [[ "${#dependency_wait_pids[@]}" -gt 0 ]]; then
  for dependency_wait_pid in "${dependency_wait_pids[@]}"; do
    wait "${dependency_wait_pid}" || true
  done
fi

# ---------------------------------------------------------------------------
# Step 4: Optional warmup
# ---------------------------------------------------------------------------
echo
echo "[distributed] Waiting for nodes to be ready..."
if [[ "${warmup_flag}" -eq 1 ]]; then
  echo "[distributed] Warming up camera daemon (first capture triggers D435 init)..."
  "${project_root}/scripts/ros2_system.sh" service call /camera_server/capture robot_grasp_msgs/srv/CaptureScene \
    "{run_id: 'warmup', depth_fusion_frames: 1, pointcloud_filter_mode: 'bilateral', pointcloud_backend: 'sdk'}" \
    >/dev/null 2>&1 || echo "[distributed] camera warmup skipped (service not ready yet)"

  echo "[distributed] Warming up vision daemon (loads YOLOv8-seg + GraspNet before navigation handoff)..."
  warmup_start="$(date +%s)"
  # Do not let an unavailable accelerator/model daemon leave the whole stack
  # stuck in "starting" forever.  A later real request can still initialize it.
  timeout 120 "${project_root}/scripts/ros2_system.sh" service call /vision_worker/warmup std_srvs/srv/Trigger \
    "{}" \
    >/dev/null 2>&1 || echo "[distributed] vision warmup skipped (service not ready yet)"
  warmup_end="$(date +%s)"
  warmup_elapsed=$((warmup_end - warmup_start))
  echo "[distributed] Vision daemon warmup completed in ${warmup_elapsed}s"
else
  echo "[distributed] Warmup skipped. First real run will initialize camera/vision daemons."
fi

cat <<EOF

Distributed stack is running and warmed up.
  session logs: ${session_root}
  ros2 cli: ${project_root}/scripts/ros2_system.sh

Recommended next commands:
  ${project_root}/scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
  ${project_root}/scripts/run_pipeline_service.sh cup
  ${project_root}/scripts/ros2_system.sh topic echo /grasp_pipeline/status
  ${project_root}/scripts/ros2_system.sh topic list | grep grasp_pipeline

If you want RViz:
  ${project_root}/scripts/ros2_system.sh topic echo /vision_worker/status
  ${project_root}/scripts/open_distributed_rviz.sh

Tip:
  /vision_worker/result_json and /grasp_pipeline/result_json now use transient-local QoS,
  so you can inspect the last run even if you subscribe after it finishes.

Press Ctrl+C here to stop all nodes together.
EOF

wait_status=0
if ! wait -n "${pids[@]}"; then
  wait_status=$?
fi

echo
echo "A node exited. Check logs under ${session_root}."
exit "${wait_status}"
