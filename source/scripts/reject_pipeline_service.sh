#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/ros2_system.sh" service call /grasp_pipeline/reject std_srvs/srv/Trigger "{}"
