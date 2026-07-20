#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
bundle_root="$(cd "${project_root}/.." && pwd)"

restore_nounset=0
if [[ $- == *u* ]]; then
  restore_nounset=1
  set +u
fi
source /opt/ros/humble/setup.bash
source "${PIPER_ROS_ROOT:-${bundle_root}/piper_ros_ws}/install/setup.bash"
if [[ "${restore_nounset}" -eq 1 ]]; then
  set -u
fi

/usr/bin/python3 - <<'PY'
import json
import math
import sys
import time

from geometry_msgs.msg import PoseArray
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


TOPICS = {
    "/grasp_pipeline/result_json": "string",
    "/vision_worker/result_json": "string",
    "/grasp_pipeline/rviz/candidate_validation_markers": "marker_array",
    "/camera_server/latest/camera_info": "camera_info",
    "/vision_worker/rviz/scene_pointcloud": "pointcloud2",
    "/vision_worker/rviz/candidate_grasps": "pose_array",
    "/vision_worker/rviz/selected_grasp": "pose_array",
    "/vision_worker/rviz/plan_waypoints": "pose_array",
    "/vision_worker/rviz/candidate_markers": "marker_array",
    "/vision_worker/rviz/selected_grasp_markers": "marker_array",
    "/vision_worker/rviz/plan_markers": "marker_array",
}


class SnapshotProbe(Node):
    def __init__(self) -> None:
        super().__init__("show_last_distributed_snapshot")
        self.messages: dict[str, object] = {}

        string_qos = QoSProfile(depth=10)
        string_qos.reliability = ReliabilityPolicy.RELIABLE
        string_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        cloud_qos = QoSProfile(depth=1)
        cloud_qos.reliability = ReliabilityPolicy.RELIABLE
        cloud_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(
            String,
            "/grasp_pipeline/result_json",
            lambda msg: self._store_string("/grasp_pipeline/result_json", msg),
            string_qos,
        )
        self.create_subscription(
            String,
            "/vision_worker/result_json",
            lambda msg: self._store_string("/vision_worker/result_json", msg),
            string_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/grasp_pipeline/rviz/candidate_validation_markers",
            lambda msg: self._store_marker_array("/grasp_pipeline/rviz/candidate_validation_markers", msg),
            cloud_qos,
        )
        self.create_subscription(
            CameraInfo,
            "/camera_server/latest/camera_info",
            lambda msg: self._store_camera_info("/camera_server/latest/camera_info", msg),
            cloud_qos,
        )
        self.create_subscription(
            PointCloud2,
            "/vision_worker/rviz/scene_pointcloud",
            lambda msg: self._store_pointcloud("/vision_worker/rviz/scene_pointcloud", msg),
            cloud_qos,
        )
        self.create_subscription(
            PoseArray,
            "/vision_worker/rviz/candidate_grasps",
            lambda msg: self._store_pose_array("/vision_worker/rviz/candidate_grasps", msg),
            cloud_qos,
        )
        self.create_subscription(
            PoseArray,
            "/vision_worker/rviz/selected_grasp",
            lambda msg: self._store_pose_array("/vision_worker/rviz/selected_grasp", msg),
            cloud_qos,
        )
        self.create_subscription(
            PoseArray,
            "/vision_worker/rviz/plan_waypoints",
            lambda msg: self._store_pose_array("/vision_worker/rviz/plan_waypoints", msg),
            cloud_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/vision_worker/rviz/candidate_markers",
            lambda msg: self._store_marker_array("/vision_worker/rviz/candidate_markers", msg),
            cloud_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/vision_worker/rviz/selected_grasp_markers",
            lambda msg: self._store_marker_array("/vision_worker/rviz/selected_grasp_markers", msg),
            cloud_qos,
        )
        self.create_subscription(
            MarkerArray,
            "/vision_worker/rviz/plan_markers",
            lambda msg: self._store_marker_array("/vision_worker/rviz/plan_markers", msg),
            cloud_qos,
        )

    def _store_string(self, topic: str, msg: String) -> None:
        try:
            self.messages[topic] = json.loads(msg.data)
        except Exception:
            self.messages[topic] = msg.data

    def _store_camera_info(self, topic: str, msg: CameraInfo) -> None:
        self.messages[topic] = {
            "frame_id": msg.header.frame_id,
            "width": int(msg.width),
            "height": int(msg.height),
            "fx": float(msg.k[0]),
            "fy": float(msg.k[4]),
            "ppx": float(msg.k[2]),
            "ppy": float(msg.k[5]),
        }

    def _store_pointcloud(self, topic: str, msg: PointCloud2) -> None:
        point_count = int(msg.width) * int(msg.height)
        approx_empty = point_count == 1 and len(msg.data) == 12
        self.messages[topic] = {
            "frame_id": msg.header.frame_id,
            "width": int(msg.width),
            "height": int(msg.height),
            "point_step": int(msg.point_step),
            "byte_size": int(len(msg.data)),
            "approx_empty": approx_empty,
        }

    def _store_pose_array(self, topic: str, msg: PoseArray) -> None:
        first_pose = msg.poses[0] if msg.poses else None
        self.messages[topic] = {
            "frame_id": msg.header.frame_id,
            "pose_count": int(len(msg.poses)),
            "first_pose": (
                {
                    "position": {
                        "x": float(first_pose.position.x),
                        "y": float(first_pose.position.y),
                        "z": float(first_pose.position.z),
                    },
                    "orientation_xyzw": {
                        "x": float(first_pose.orientation.x),
                        "y": float(first_pose.orientation.y),
                        "z": float(first_pose.orientation.z),
                        "w": float(first_pose.orientation.w),
                    },
                }
                if first_pose is not None
                else None
            ),
        }

    def _store_marker_array(self, topic: str, msg: MarkerArray) -> None:
        namespaces = sorted({str(marker.ns) for marker in msg.markers if str(marker.ns)})
        marker_types = sorted({int(marker.type) for marker in msg.markers})
        self.messages[topic] = {
            "marker_count": int(len(msg.markers)),
            "namespaces": namespaces,
            "marker_types": marker_types,
        }


def main() -> int:
    rclpy.init()
    node = SnapshotProbe()
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if len(node.messages) == len(TOPICS):
                break

        for topic, topic_kind in TOPICS.items():
            print(f"=== {topic} ===")
            payload = node.messages.get(topic)
            if payload is None:
                print("NO_MESSAGE")
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))
            if topic != list(TOPICS.keys())[-1]:
                print()
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
PY
