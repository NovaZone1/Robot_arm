from glob import glob
import os

from setuptools import find_packages, setup


package_name = "robot_grasp_ros2"


def files_only(pattern: str) -> list[str]:
    return [path for path in glob(pattern) if os.path.isfile(path)]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["src", "src.*", package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "distributed"), glob("config/distributed/*.yaml")),
        (
            os.path.join("share", package_name, "assets", "item_references"),
            files_only("assets/item_references/*"),
        ),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="wt",
    maintainer_email="wt@example.com",
    description="ROS2 wrapper package for the migrated robot grasp pipeline.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "grasp_pipeline_cli = src.run_grasp_pipeline_ros2:main",
            "grasp_pipeline_node = robot_grasp_ros2.grasp_pipeline_node:main",
            "camera_server_node = robot_grasp_ros2.camera_server_node:main",
            "vision_worker_node = robot_grasp_ros2.vision_worker_node:main",
            "robot_executor_node = robot_grasp_ros2.robot_executor_node:main",
            "pipeline_orchestrator_node = robot_grasp_ros2.pipeline_orchestrator_node:main",
            "scout_scan_controller_node = robot_grasp_ros2.scout_scan_controller_node:main",
            "red_flag_start_gate = robot_grasp_ros2.red_flag_start_node:main",
            "piper_interactive_marker_node = robot_grasp_ros2.piper_interactive_marker_node:main",
            "piper_pose_bridge_node = robot_grasp_ros2.piper_pose_bridge_node:main",
            "joint_state_feedback_relay_node = robot_grasp_ros2.joint_state_feedback_relay_node:main",
        ],
    },
)
