import numpy as np

from robot_grasp_ros2.rviz_visualization import build_candidate_validation_marker_array


def test_build_candidate_validation_marker_array_marks_rejected_and_selected_candidates():
    markers = build_candidate_validation_marker_array(
        validation_records=[
            {
                "candidate_index": 0,
                "candidate_score": 0.91,
                "translation_camera_m": [0.1, 0.2, 0.3],
                "selection_result": "rejected_by_robot_validation",
                "robot_validation_stage": "grasp",
                "ik_error_type": "timeout",
            },
            {
                "candidate_index": 1,
                "candidate_score": 0.77,
                "translation_camera_m": [0.2, 0.1, 0.4],
                "selection_result": "selected_for_execution",
                "robot_validation_result": "accepted",
            },
        ],
        camera_frame="camera_color_optical_frame",
        stamp=None,
    )

    texts = [marker.text for marker in markers.markers if getattr(marker, "text", "")]
    namespaces = {marker.ns for marker in markers.markers}

    assert "candidate_validation" in namespaces
    assert any("cand0 rejected@grasp timeout" in text for text in texts)
    assert any("cand1 selected fallback" in text for text in texts)


def test_build_candidate_validation_marker_array_returns_delete_all_when_empty():
    markers = build_candidate_validation_marker_array(
        validation_records=[],
        camera_frame="camera_color_optical_frame",
        stamp=None,
    )

    assert len(markers.markers) == 1
    assert markers.markers[0].action == markers.markers[0].DELETEALL
