#!/usr/bin/env bash
# Bridge a completed Nav2 unloading-point arrival into target-box alignment and
# the calibrated fixed-TCP placement sequence. The target identity is the one
# freshly resolved from the photo card by the preceding grasp run.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ros2_cli="${ROBOT_GRASP_ROS2_CLI:-${script_dir}/ros2_system.sh}"
pipeline_node="/grasp_pipeline"
preflight_only=false

usage() {
  cat >&2 <<'EOF'
Usage:
  run_navigation_place_handoff.sh
  run_navigation_place_handoff.sh --preflight

The normal run reads target_item_id from /grasp_pipeline. It does not accept a
manual item argument, preventing the unloading target from diverging from the
photo-card target used for the preceding grasp.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight)
      preflight_only=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

node_output="$(${ros2_cli} node list 2>/dev/null)"
node_count="$(grep -Fxc "${pipeline_node}" <<<"${node_output}" || true)"
if [[ "${node_count}" -ne 1 ]]; then
  echo "place handoff blocked: expected exactly one ${pipeline_node}, found ${node_count}" >&2
  exit 20
fi

probe_output="$(${ros2_cli} service call \
  /grasp_pipeline/probe std_srvs/srv/Trigger "{}" 2>&1)"
if ! grep -Eqi 'success[=:][[:space:]]*true' <<<"${probe_output}"; then
  echo "place handoff blocked: grasp pipeline probe failed" >&2
  echo "${probe_output}" >&2
  exit 21
fi

service_output="$(${ros2_cli} service list 2>/dev/null || true)"
for required_service in \
  /base_scan_controller/move_relative \
  /grasp_pipeline/scan_and_align_placement_target \
  /grasp_pipeline/execute_aligned_place; do
  if ! grep -Fxq "${required_service}" <<<"${service_output}"; then
    echo "place handoff blocked: ${required_service} is unavailable" >&2
    exit 22
  fi
done

odom_info="$(${ros2_cli} topic info /odom 2>&1 || true)"
odom_publishers="$(sed -nE 's/^Publisher count:[[:space:]]*([0-9]+).*$/\1/p' <<<"${odom_info}" | tail -1)"
if [[ -z "${odom_publishers}" || "${odom_publishers}" -lt 1 ]]; then
  echo "place handoff blocked: /odom has no publisher" >&2
  echo "${odom_info}" >&2
  exit 23
fi

echo "place handoff ready: services=ok odom_publishers=${odom_publishers}"
if [[ "${preflight_only}" == true ]]; then
  exit 0
fi

target_output="$(timeout 12 "${ros2_cli}" param get "${pipeline_node}" target_item_id 2>&1 || true)"
target_item_id="$(sed -nE \
  's/^(String value is:|value:)[[:space:]]*//p' <<<"${target_output}" \
  | tail -1 | tr -d "'\"" | xargs)"

# A ros2 CLI parameter query can occasionally wedge in DDS discovery even
# though the pipeline node and its services are healthy. The grasp result is
# already persisted atomically before navigation resumes, so use that fresh,
# successful artifact as the authoritative fallback instead of leaving the
# vehicle parked forever without starting the placement scan.
if [[ ! "${target_item_id}" =~ ^(red_block|yellow_block|blue_block|orange_bottle|dark_bottle|green_bottle)$ ]]; then
  target_item_id="$(python3 - /home/nvidia/auto/Robot_arm/log/distributed_runs/latest_run.txt <<'PY'
import json
import pathlib
import sys
import time

pointer = pathlib.Path(sys.argv[1])
try:
    run_dir = pathlib.Path(pointer.read_text(encoding="utf-8").strip())
    result_path = run_dir / "final_result.json"
    if time.time() - result_path.stat().st_mtime > 7200.0:
        raise RuntimeError("latest successful grasp is older than two hours")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    execution = payload.get("execution") or {}
    item = payload.get("target_item") or {}
    if payload.get("status") != "ok" or execution.get("status") != "ok":
        raise RuntimeError("latest grasp did not execute successfully")
    print(str(item.get("item_id") or "").strip())
except Exception:
    print("")
PY
)"
  if [[ -n "${target_item_id}" ]]; then
    echo "place handoff: parameter query unavailable; recovered target=${target_item_id} from latest successful grasp"
  fi
fi
case "${target_item_id}" in
  red_block|yellow_block|blue_block|orange_bottle|dark_bottle|green_bottle)
    ;;
  *)
    echo "place handoff blocked: no fresh catalog target is available: '${target_item_id}'" >&2
    exit 24
    ;;
esac

set_param() {
  local name="$1"
  local value="$2"
  timeout 12 "${ros2_cli}" param set "${pipeline_node}" "${name}" "${value}" >/dev/null
}

service_succeeded() {
  grep -Eqi 'success[=:][[:space:]]*true'
}

reset_safety_latches() {
  set_param base_target_alignment_enabled false >/dev/null 2>&1 || true
  set_param base_aligned_place_enabled false >/dev/null 2>&1 || true
}
trap reset_safety_latches EXIT INT TERM

echo "place handoff: scanning and aligning box for ${target_item_id}"
set_param base_target_alignment_enabled true
align_output="$(${ros2_cli} service call \
  /grasp_pipeline/scan_and_align_placement_target \
  std_srvs/srv/Trigger "{}" 2>&1)"
set_param base_target_alignment_enabled false
if ! service_succeeded <<<"${align_output}"; then
  echo "place handoff failed: target-box scan/alignment did not complete" >&2
  echo "${align_output}" >&2
  exit 25
fi

echo "place handoff: box aligned; executing calibrated release for ${target_item_id}"
set_param base_aligned_place_enabled true
place_output="$(${ros2_cli} service call \
  /grasp_pipeline/execute_aligned_place \
  std_srvs/srv/Trigger "{}" 2>&1)"
set_param base_aligned_place_enabled false
if ! service_succeeded <<<"${place_output}"; then
  echo "place handoff failed: calibrated placement did not complete" >&2
  echo "${place_output}" >&2
  exit 26
fi

trap - EXIT INT TERM
reset_safety_latches
echo "place handoff completed: target=${target_item_id}"
