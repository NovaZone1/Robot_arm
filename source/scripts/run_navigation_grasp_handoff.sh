#!/usr/bin/env bash
# Bridge a completed Nav2 grasp-point arrival into the distributed arm grasp
# pipeline. This script never sends a Nav2 goal; the caller must invoke it only
# after navigation has reported STATUS_SUCCEEDED.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
ros2_cli="${ROBOT_GRASP_ROS2_CLI:-${script_dir}/ros2_system.sh}"
artifact_root="${ROBOT_GRASP_ARTIFACT_ROOT:-$(cd "${project_root}/.." && pwd)/log/distributed_runs}"
pipeline_node="/grasp_pipeline"
target_item_id=""
prompt=""
preflight_only=false
wait_timeout_s=900

usage() {
  cat >&2 <<'EOF'
Usage:
  run_navigation_grasp_handoff.sh
  run_navigation_grasp_handoff.sh --target red_block [--prompt "red block"]
  run_navigation_grasp_handoff.sh --preflight

Canonical targets:
  red_block yellow_block blue_block orange_bottle dark_bottle green_bottle

By default the target is recognized from the printed photo card after the arm
reaches its observation pose. --target is retained only as an explicit manual
diagnostic fallback. The distributed grasp stack must already be running.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || usage
      target_item_id="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -ge 2 ]] || usage
      prompt="$2"
      shift 2
      ;;
    --preflight)
      preflight_only=true
      shift
      ;;
    --wait-timeout)
      [[ $# -ge 2 ]] || usage
      wait_timeout_s="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ -z "${target_item_id}" ]]; then
        target_item_id="$1"
        shift
      else
        usage
      fi
      ;;
  esac
done

node_output="$(${ros2_cli} node list 2>/dev/null)"
node_count="$(grep -Fxc "${pipeline_node}" <<<"${node_output}" || true)"
if [[ "${node_count}" -ne 1 ]]; then
  echo "handoff blocked: expected exactly one ${pipeline_node}, found ${node_count}" >&2
  exit 10
fi

case "${target_item_id}" in
  "")             default_prompt="" ;;
  red_block)      default_prompt="red block" ;;
  yellow_block)   default_prompt="yellow block" ;;
  blue_block)     default_prompt="blue block" ;;
  orange_bottle)  default_prompt="orange bottle" ;;
  dark_bottle)    default_prompt="dark bottle" ;;
  green_bottle)   default_prompt="green bottle" ;;
  *)
    echo "handoff blocked: invalid or empty target_item_id='${target_item_id}'" >&2
    exit 11
    ;;
esac
prompt="${prompt:-${default_prompt}}"

probe_output="$(${ros2_cli} service call \
  /grasp_pipeline/probe std_srvs/srv/Trigger "{}" 2>&1)"
if ! grep -Eqi 'success[=:][[:space:]]*true' <<<"${probe_output}"; then
  echo "handoff blocked: grasp pipeline probe failed" >&2
  echo "${probe_output}" >&2
  exit 12
fi

service_output="$(${ros2_cli} service list 2>/dev/null || true)"
if ! grep -Fxq '/base_scan_controller/move_relative' <<<"${service_output}"; then
  echo "handoff blocked: /base_scan_controller/move_relative is unavailable" >&2
  exit 13
fi

odom_info="$(${ros2_cli} topic info /odom 2>&1 || true)"
odom_publishers="$(sed -nE 's/^Publisher count:[[:space:]]*([0-9]+).*$/\1/p' <<<"${odom_info}" | tail -1)"
if [[ -z "${odom_publishers}" || "${odom_publishers}" -lt 1 ]]; then
  echo "handoff blocked: /odom has no publisher" >&2
  echo "${odom_info}" >&2
  exit 14
fi

if [[ -z "${target_item_id}" ]]; then
  echo "handoff ready: target=photo_card(auto) odom_publishers=${odom_publishers}"
else
  echo "handoff ready: target=${target_item_id} prompt='${prompt}' odom_publishers=${odom_publishers}"
fi
if [[ "${preflight_only}" == true ]]; then
  exit 0
fi

set_param() {
  local name="$1"
  local value="$2"
  ${ros2_cli} param set "${pipeline_node}" "${name}" "${value}" >/dev/null
}

if [[ -z "${target_item_id}" ]]; then
  set_param auto_target_from_card true
  set_param target_item_id "''"
  set_param prompt "''"
else
  set_param auto_target_from_card false
  set_param target_item_id "${target_item_id}"
  set_param prompt "${prompt}"
fi
set_param execute true
set_param confirm false
# This bridge owns pickup only. Placement is performed after the route reaches
# its dedicated unloading point, so a stale Dashboard checkbox must not cause
# an early release at the pickup table.
set_param place_after_grasp false
set_param base_grasp_scan_enabled true
set_param move_to_placement_observation_after_grasp true

before_run="$(find "${artifact_root}" -maxdepth 1 -type d -name 'grasp-*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
trigger_output="$(${ros2_cli} service call \
  /grasp_pipeline/run std_srvs/srv/Trigger "{}" 2>&1)"
if ! grep -Eqi 'success[=:][[:space:]]*true' <<<"${trigger_output}"; then
  echo "handoff failed: grasp run was not accepted" >&2
  echo "${trigger_output}" >&2
  exit 15
fi
echo "navigation handoff accepted by grasp pipeline"

deadline=$(( $(date +%s) + wait_timeout_s ))
run_dir=""
while [[ $(date +%s) -lt ${deadline} ]]; do
  candidate="$(find "${artifact_root}" -maxdepth 1 -type d -name 'grasp-*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "${candidate}" && "${candidate}" != "${before_run}" ]]; then
    run_dir="${candidate}"
    break
  fi
  sleep 1
done
if [[ -z "${run_dir}" ]]; then
  echo "handoff failed: no new grasp run artifact appeared" >&2
  exit 16
fi

stop_grasp_on_interrupt() {
  ${ros2_cli} service call /grasp_pipeline/stop std_srvs/srv/Trigger "{}" \
    >/dev/null 2>&1 || true
}
trap stop_grasp_on_interrupt INT TERM

final_result="${run_dir}/final_result.json"
while [[ $(date +%s) -lt ${deadline} ]]; do
  if [[ -s "${final_result}" ]]; then
    status="$(jq -r '.status // empty' "${final_result}" 2>/dev/null || true)"
    case "${status}" in
      ok|completed)
        echo "grasp handoff completed: run=$(basename "${run_dir}") status=${status}"
        exit 0
        ;;
      failed|no_candidate|stopped|cancelled|rejected)
        summary="$(jq -r '.summary // .message // "no summary"' "${final_result}" 2>/dev/null || true)"
        echo "grasp handoff failed: run=$(basename "${run_dir}") status=${status}" >&2
        echo "${summary}" >&2
        exit 17
        ;;
    esac
  fi
  sleep 1
done

echo "grasp handoff timed out after ${wait_timeout_s}s; sending stop" >&2
stop_grasp_on_interrupt
exit 18
