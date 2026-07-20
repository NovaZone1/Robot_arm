from __future__ import annotations

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import FindExecutable, LaunchConfiguration
from launch_ros.actions import Node

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ROS_ENV_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "ros_env_graspnet.sh")


def _display_nodes(context, *_args, **_kwargs):
    with_display = LaunchConfiguration("with_display").perform(context).lower() == "true"
    if not with_display:
        return []

    import xacro

    end_effector = LaunchConfiguration("end_effector").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    agx_share = get_package_share_directory("agx_arm_description")
    xacro_file = os.path.join(agx_share, "urdf", "agx_arm_description.urdf.xacro")
    if not rviz_config:
        rviz_config = os.path.join(agx_share, "rviz", "default.rviz")

    robot_description = xacro.process_file(
        xacro_file,
        mappings={
            "arm_type": "piper",
            "end_effector": end_effector,
            "with_camera_stand": "false",
            "with_camera": "false",
        },
    ).toxml()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="piper_robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="piper_interactive_rviz",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ]


def _source_nodes(context, *_args, **_kwargs):
    marker_frame_id = LaunchConfiguration("marker_frame_id").perform(context)
    marker_ee_link = LaunchConfiguration("marker_ee_link").perform(context)
    robot_backend = LaunchConfiguration("robot_backend").perform(context)
    auto_enable = LaunchConfiguration("auto_enable").perform(context)
    speed_percent = LaunchConfiguration("speed_percent").perform(context)
    with_joint_feedback_relay = (
        LaunchConfiguration("with_joint_feedback_relay").perform(context).lower() == "true"
    )

    actions = [
        ExecuteProcess(
            cmd=[
                "bash",
                "-lc",
                " ".join(
                    [
                        f"source {shlex.quote(ROS_ENV_SCRIPT)} >/dev/null",
                        "&&",
                        "python -m robot_grasp_ros2.piper_interactive_marker_node --ros-args",
                        f"-p frame_id:={shlex.quote(marker_frame_id)}",
                        f"-p ee_link:={shlex.quote(marker_ee_link)}",
                    ]
                ),
            ],
            name="piper_interactive_marker",
            output="screen",
            cwd=PROJECT_ROOT,
        ),
        ExecuteProcess(
            cmd=[
                "bash",
                "-lc",
                " ".join(
                    [
                        f"source {shlex.quote(ROS_ENV_SCRIPT)} >/dev/null",
                        "&&",
                        "python -m robot_grasp_ros2.piper_pose_bridge_node --ros-args",
                        f"-p robot_backend:={shlex.quote(robot_backend)}",
                        f"-p auto_enable:={shlex.quote(auto_enable)}",
                        f"-p command_frame:={shlex.quote(marker_frame_id)}",
                        f"-p speed_percent:={shlex.quote(speed_percent)}",
                    ]
                ),
            ],
            name="piper_pose_bridge",
            output="screen",
            cwd=PROJECT_ROOT,
        ),
    ]

    if with_joint_feedback_relay:
        actions.append(
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    " ".join(
                        [
                            f"source {shlex.quote(ROS_ENV_SCRIPT)} >/dev/null",
                            "&&",
                            "python -m robot_grasp_ros2.joint_state_feedback_relay_node",
                        ]
                    ),
                ],
                name="joint_state_feedback_relay",
                output="screen",
                cwd=PROJECT_ROOT,
            )
        )

    return actions


def generate_launch_description():
    robot_backend = LaunchConfiguration("robot_backend")
    auto_enable = LaunchConfiguration("auto_enable")
    speed_percent = LaunchConfiguration("speed_percent")
    with_display = LaunchConfiguration("with_display")
    with_joint_feedback_relay = LaunchConfiguration("with_joint_feedback_relay")
    marker_frame_id = LaunchConfiguration("marker_frame_id")
    marker_ee_link = LaunchConfiguration("marker_ee_link")
    end_effector = LaunchConfiguration("end_effector")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_backend", default_value="fake"),
            DeclareLaunchArgument("auto_enable", default_value="true"),
            DeclareLaunchArgument("speed_percent", default_value="25.0"),
            DeclareLaunchArgument("with_display", default_value="false"),
            DeclareLaunchArgument("with_joint_feedback_relay", default_value="true"),
            DeclareLaunchArgument("marker_frame_id", default_value="base_link"),
            DeclareLaunchArgument("marker_ee_link", default_value="gripper_base"),
            DeclareLaunchArgument("end_effector", default_value="gripper"),
            DeclareLaunchArgument("rviz_config", default_value=""),
            OpaqueFunction(function=_display_nodes),
            OpaqueFunction(function=_source_nodes),
        ]
    )
