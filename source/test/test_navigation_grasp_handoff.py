import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = PROJECT_ROOT / "scripts" / "run_navigation_grasp_handoff.sh"
PLACE_HANDOFF = PROJECT_ROOT / "scripts" / "run_navigation_place_handoff.sh"
ROUTE = Path(
    "/home/nvidia/auto/ROS2_FOR_SCOUT_MINI/my_party/navigation_ws/src/"
    "scout_navigation_bringup/scripts/run_indoor_recorded_route.sh"
)
INDOOR03_ROUTE = Path(
    "/home/nvidia/auto/ROS2_FOR_SCOUT_MINI/my_party/navigation_ws/src/"
    "scout_navigation_bringup/scripts/run_indoor03_recorded_route.sh"
)


def _mock_ros2_cli(tmp_path: Path) -> Path:
    script = tmp_path / "mock_ros2.sh"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case \"$1 $2\" in
  \"node list\")
    echo /grasp_pipeline
    ;;
  \"param get\")
    echo 'String value is: red_block'
    ;;
  \"param set\")
    echo 'Set parameter successful'
    ;;
  \"service list\")
    echo /base_scan_controller/move_relative
    echo /grasp_pipeline/probe
    echo /grasp_pipeline/run
    echo /grasp_pipeline/scan_and_align_placement_target
    echo /grasp_pipeline/execute_aligned_place
    if [[ \"${MOCK_LONG_SERVICE_LIST:-0}\" == 1 ]]; then
      for index in $(seq 1 20000); do
        echo \"/mock_service_${index}\"
      done
    fi
    ;;
  \"service call\")
    if [[ \"${3:-}\" == /grasp_pipeline/run ]]; then
      run_dir=\"${MOCK_ARTIFACT_ROOT}/grasp-mock\"
      mkdir -p \"${run_dir}\"
      printf '%s\\n' '{\"status\":\"ok\",\"summary\":\"mock completed\"}' \\
        > \"${run_dir}/final_result.json\"
    fi
    echo \"response: std_srvs.srv.Trigger_Response(success=True, message='ok')\"
    ;;
  \"topic info\")
    echo 'Type: nav_msgs/msg/Odometry'
    echo 'Publisher count: 1'
    ;;
  *)
    echo \"unexpected mock ros2 invocation: $*\" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_navigation_handoff_preflight_uses_automatic_photo_card_target(tmp_path):
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    mock = _mock_ros2_cli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "ROBOT_GRASP_ROS2_CLI": str(mock),
            "ROBOT_GRASP_ARTIFACT_ROOT": str(artifact_root),
            "MOCK_ARTIFACT_ROOT": str(artifact_root),
        }
    )

    result = subprocess.run(
        [str(HANDOFF), "--preflight"],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "target=photo_card(auto)" in result.stdout
    assert "odom_publishers=1" in result.stdout


def test_navigation_handoff_preflight_handles_long_service_list_with_pipefail(
    tmp_path,
):
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    mock = _mock_ros2_cli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "ROBOT_GRASP_ROS2_CLI": str(mock),
            "ROBOT_GRASP_ARTIFACT_ROOT": str(artifact_root),
            "MOCK_ARTIFACT_ROOT": str(artifact_root),
            "MOCK_LONG_SERVICE_LIST": "1",
        }
    )

    result = subprocess.run(
        [str(HANDOFF), "--preflight", "--target", "red_block"],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "handoff ready" in result.stdout


def test_navigation_handoff_triggers_and_waits_for_grasp_result(tmp_path):
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    mock = _mock_ros2_cli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "ROBOT_GRASP_ROS2_CLI": str(mock),
            "ROBOT_GRASP_ARTIFACT_ROOT": str(artifact_root),
            "MOCK_ARTIFACT_ROOT": str(artifact_root),
        }
    )

    result = subprocess.run(
        [str(HANDOFF), "--target", "blue_block", "--wait-timeout", "5"],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "navigation handoff accepted" in result.stdout
    assert "grasp handoff completed" in result.stdout


def test_navigation_place_handoff_aligns_then_releases_same_target(tmp_path):
    mock = _mock_ros2_cli(tmp_path)
    env = os.environ.copy()
    env["ROBOT_GRASP_ROS2_CLI"] = str(mock)

    result = subprocess.run(
        [str(PLACE_HANDOFF)],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "scanning and aligning box for red_block" in result.stdout
    assert "executing calibrated release for red_block" in result.stdout
    assert "place handoff completed: target=red_block" in result.stdout


def test_navigation_route_preflights_before_goals_and_handoffs_afterward():
    text = ROUTE.read_text(encoding="utf-8")
    preflight = text.index('"${grasp_handoff}" --preflight')
    red_flag_gate = text.index('"${red_flag_start_gate}"')
    navigation = text.index(
        "ros2 run scout_navigation_bringup dock_to_indoor_grasp_point.py"
    )
    handoff = text.index('"${grasp_handoff}" "${handoff_args[@]}"')

    assert preflight < red_flag_gate < navigation < handoff


def test_navigation_route_uses_docking_sequence_and_red_flag_gate_by_default():
    text = ROUTE.read_text(encoding="utf-8")

    assert "ros2 run scout_navigation_bringup dock_to_indoor_grasp_point.py" in text
    assert '"${RED_FLAG_START_ENABLED:-1}" == "1"' in text
    assert "RED_FLAG_START_ENABLED=0" in text


def test_indoor03_route_preflights_then_hands_off_after_pickup_docking():
    text = INDOOR03_ROUTE.read_text(encoding="utf-8")

    preflight = text.index('"${grasp_handoff}" --preflight')
    red_flag_gate = text.index('"${red_flag_start_gate}"')
    first_transit = text.index('go_normal "门口（红绿灯）"')
    pickup = text.index('dock_slow "取件"')
    handoff = text.index('"${grasp_handoff}" "${handoff_args[@]}"')
    next_transit = text.index('go_normal "另一侧1"')

    assert preflight < red_flag_gate < first_transit < pickup < handoff < next_transit
    assert "handoff_args=()" in text
    assert "--target" not in text


def test_indoor03_route_uses_red_flag_gate_by_default():
    text = INDOOR03_ROUTE.read_text(encoding="utf-8")

    assert 'red_flag_start_gate="/home/nvidia/auto/Robot_arm/source/scripts/wait_for_red_flag_start.sh"' in text
    assert '"${RED_FLAG_START_ENABLED:-1}" == "1"' in text
    assert "RED_FLAG_START_ENABLED=0" in text


def test_red_flag_gate_retries_same_verified_pose_only_once():
    text = (
        PROJECT_ROOT / "scripts" / "wait_for_red_flag_start.sh"
    ).read_text(encoding="utf-8")

    assert 'RED_FLAG_MOVE_ATTEMPTS:-2' in text
    assert "RED_FLAG_MOVE_ATTEMPTS must be 1 or 2" in text
    assert "/enable_srv piper_msgs/srv/Enable" in text
    assert text.count("x_mm: 44.300") == 1
    assert "home_after_red_flag" in text
    assert "nohup" in text


def test_indoor03_route_hands_off_placement_after_unloading_docking():
    text = INDOOR03_ROUTE.read_text(encoding="utf-8")

    place_preflight = text.index('"${place_handoff}" --preflight')
    first_transit = text.index('go_normal "门口（红绿灯）"')
    unload = text.index('dock_slow "放置"')
    place_handoff = text.index('"${place_handoff}"', unload)
    final_transit = text.index('go_normal "另一侧2"')

    assert place_preflight < first_transit < unload < place_handoff < final_transit
