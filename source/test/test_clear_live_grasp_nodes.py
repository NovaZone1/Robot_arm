from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_grasp_ros2.clear_live_grasp_nodes import (
    CleanupTarget,
    build_cleanup_targets,
    parse_pgrep_lines,
)


def test_build_cleanup_targets_covers_live_grasp_stack():
    targets = build_cleanup_targets(PROJECT_ROOT)

    assert [target.name for target in targets] == [
        "distributed_stack",
        "moveit_ik",
        "piper_driver",
        "distributed_rviz",
    ]
    assert any("pipeline_orchestrator_node" in target.pattern for target in targets)
    assert any("move_group" in target.pattern for target in targets)
    assert any("piper_single_ctrl" in target.pattern for target in targets)
    assert any("distributed_grasp_pipeline" in target.pattern and "rviz" in target.pattern for target in targets)


def test_parse_pgrep_lines_groups_matching_processes_and_skips_current_pid():
    targets = [
        CleanupTarget("distributed_stack", r"robot_grasp_ros2\.(camera_server_node|pipeline_orchestrator_node)"),
        CleanupTarget("moveit_ik", r"move_group"),
    ]
    pgrep_lines = [
        "101 /usr/bin/python3 -m robot_grasp_ros2.camera_server_node --ros-args",
        "102 /opt/ros/jazzy/lib/moveit_ros_move_group/move_group --ros-args",
        "999 /usr/bin/python3 -m robot_grasp_ros2.pipeline_orchestrator_node --ros-args",
        "bad-line-without-pid",
    ]

    grouped = parse_pgrep_lines(targets, pgrep_lines, current_pid=999)

    assert [process.pid for process in grouped["distributed_stack"]] == [101]
    assert [process.pid for process in grouped["moveit_ik"]] == [102]
