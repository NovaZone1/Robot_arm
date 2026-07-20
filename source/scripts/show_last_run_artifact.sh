#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
artifact_root="${project_root}/../log/distributed_runs"
latest_file="${artifact_root}/latest_run.txt"

if [[ ! -f "${latest_file}" ]]; then
  echo "No distributed run artifact found under ${artifact_root}" >&2
  exit 1
fi

artifact_dir="$(head -n 1 "${latest_file}" | tr -d '\r')"
if [[ -z "${artifact_dir}" || ! -d "${artifact_dir}" ]]; then
  echo "Latest artifact directory is missing: ${artifact_dir}" >&2
  exit 1
fi

echo "artifact_dir=${artifact_dir}"
echo

for file in request.json cycles.json final_result.json; do
  path="${artifact_dir}/${file}"
  echo "=== ${file} ==="
  if [[ -f "${path}" ]]; then
    sed -n '1,240p' "${path}"
  else
    echo "MISSING"
  fi
  echo
done
