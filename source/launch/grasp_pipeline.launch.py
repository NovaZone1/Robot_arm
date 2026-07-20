from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    prompt = LaunchConfiguration("prompt")
    robot_backend = LaunchConfiguration("robot_backend")
    auto_start = LaunchConfiguration("auto_start")
    execute = LaunchConfiguration("execute")
    show_pointcloud = LaunchConfiguration("show_pointcloud")
    precenter = LaunchConfiguration("precenter")
    can = LaunchConfiguration("can")

    return LaunchDescription(
        [
            DeclareLaunchArgument("prompt", default_value=""),
            DeclareLaunchArgument("robot_backend", default_value="fake"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("show_pointcloud", default_value="false"),
            DeclareLaunchArgument("precenter", default_value="false"),
            DeclareLaunchArgument("can", default_value="can0"),
            Node(
                package="robot_grasp_ros2",
                executable="grasp_pipeline_node",
                name="grasp_pipeline",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("robot_grasp_ros2"), "config", "grasp_pipeline.params.yaml"]
                    ),
                    {
                        "prompt": ParameterValue(prompt, value_type=str),
                        "robot_backend": ParameterValue(robot_backend, value_type=str),
                        "auto_start": ParameterValue(auto_start, value_type=bool),
                        "execute": ParameterValue(execute, value_type=bool),
                        "show_pointcloud": ParameterValue(show_pointcloud, value_type=bool),
                        "precenter": ParameterValue(precenter, value_type=bool),
                        "can": ParameterValue(can, value_type=str),
                    },
                ],
            ),
        ]
    )
