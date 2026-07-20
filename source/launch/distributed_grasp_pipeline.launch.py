from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("robot_grasp_ros2")
    distributed_config = PathJoinSubstitution([package_share, "config", "distributed"])

    return LaunchDescription(
        [
            Node(
                package="robot_grasp_ros2",
                executable="camera_server_node",
                name="camera_server",
                output="screen",
                parameters=[PathJoinSubstitution([distributed_config, "camera_server.params.yaml"])],
            ),
            Node(
                package="robot_grasp_ros2",
                executable="vision_worker_node",
                name="vision_worker",
                output="screen",
                parameters=[PathJoinSubstitution([distributed_config, "vision_worker.params.yaml"])],
            ),
            Node(
                package="robot_grasp_ros2",
                executable="robot_executor_node",
                name="robot_executor",
                output="screen",
                parameters=[PathJoinSubstitution([distributed_config, "robot_executor.params.yaml"])],
            ),
            Node(
                package="robot_grasp_ros2",
                executable="pipeline_orchestrator_node",
                name="grasp_pipeline",
                output="screen",
                parameters=[PathJoinSubstitution([distributed_config, "pipeline_orchestrator.params.yaml"])],
            ),
        ]
    )
