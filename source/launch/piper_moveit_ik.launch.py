from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_param_builder import load_yaml


def _resolve_piper_ros_root() -> Path:
    env_root = os.environ.get("PIPER_ROS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    package_share = Path(get_package_share_directory("robot_grasp_ros2")).resolve()
    candidates = [
        package_share.parents[4] / "piper_ros_ws",
        Path.home() / "piper_grasp_project" / "piper_ros_ws",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_move_group_parameters() -> list[dict[str, object]]:
    package_share = Path(get_package_share_directory("robot_grasp_ros2")).resolve()
    piper_root = _resolve_piper_ros_root()
    piper_description_urdf = piper_root / "src" / "piper_ros" / "src" / "piper_description" / "urdf" / "piper_description.urdf"
    moveit_config_root = piper_root / "src" / "piper_ros" / "src" / "piper_moveit" / "piper_with_gripper_moveit" / "config"
    default_config_root = Path(get_package_share_directory("moveit_configs_utils")) / "default_configs"
    override_joint_limits = package_share / "config" / "piper_moveit_ik_joint_limits.yaml"

    ompl_config = load_yaml(default_config_root / "ompl_planning.yaml")
    ompl_config.update(load_yaml(default_config_root / "ompl_defaults.yaml"))

    return [
        {"robot_description": piper_description_urdf.read_text()},
        {"robot_description_semantic": (moveit_config_root / "piper.srdf").read_text()},
        {"robot_description_kinematics": load_yaml(moveit_config_root / "kinematics.yaml")},
        {"robot_description_planning": load_yaml(override_joint_limits)},
        {
            "planning_pipelines": ["ompl"],
            "default_planning_pipeline": "ompl",
            "ompl": ompl_config,
        },
        {
            "allow_trajectory_execution": False,
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
            "publish_robot_description": True,
            "publish_robot_description_semantic": True,
            "monitor_dynamics": False,
        },
    ]


def generate_launch_description() -> LaunchDescription:
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=_load_move_group_parameters(),
    )
    return LaunchDescription([move_group])
