#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ros2_cli="${script_dir}/ros2_system.sh"
node_name="/grasp_pipeline"
backend=""

usage() {
  echo "Usage: $(basename "$0") <prompt> [--robot-backend fake|ros2]" >&2
  echo "  note: --robot-backend is only applied when /grasp_pipeline exposes that parameter" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

prompt="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-backend)
      [[ $# -ge 2 ]] || usage
      backend="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

node_count="$("${ros2_cli}" node list 2>/dev/null | grep -Fx "${node_name}" | wc -l | tr -d ' ')"
if [[ "${node_count}" -eq 0 ]]; then
  echo "No ${node_name} node is running. Start it first with either:" >&2
  echo "  ./scripts/run_grasp_pipeline_node_graspnet.sh" >&2
  echo "  ./scripts/run_distributed_stack_graspnet.sh" >&2
  exit 1
fi

if [[ "${node_count}" -ne 1 ]]; then
  echo "Found ${node_count} ${node_name} nodes. Service routing is ambiguous; keep exactly one node before calling /run." >&2
  pgrep -af "robot_grasp_ros2.grasp_pipeline_node|grasp_pipeline_node" || true
  exit 1
fi

if [[ -n "${backend}" ]]; then
  if "${ros2_cli}" param get "${node_name}" robot_backend >/dev/null 2>&1; then
    "${ros2_cli}" param set "${node_name}" robot_backend "${backend}" >/dev/null
  else
    echo "warning: ${node_name} does not expose robot_backend; skipping backend override" >&2
  fi
fi
"${ros2_cli}" param set "${node_name}" prompt "${prompt}" >/dev/null
"${ros2_cli}" service call "${node_name}/run" std_srvs/srv/Trigger "{}"
