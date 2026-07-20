from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_distributed_rviz_enables_generated_and_execution_pose_displays_by_default():
    rviz_text = (PROJECT_ROOT / "rviz" / "distributed_grasp_pipeline.rviz").read_text(
        encoding="utf-8"
    )

    assert "Name: Candidate Grasps" in rviz_text
    assert "Name: Selected Grasp Pose" in rviz_text
    assert "Name: Plan Waypoints" in rviz_text
    assert "Name: Plan Markers" in rviz_text
    assert "Name: Candidate Validation Markers" in rviz_text

    assert "Name: Candidate Grasps\n      Shaft Length: 0.05000000074505806\n      Shaft Radius: 0.003000000026077032\n      Shape: Arrow\n      Topic:" in rviz_text
    assert "Name: Selected Grasp Pose\n      Shaft Length: 0.06499999761581421\n      Shaft Radius: 0.004000000189989805\n      Shape: Arrow\n      Topic:" in rviz_text
    assert "Name: Plan Waypoints\n      Shaft Length: 0.05999999865889549\n      Shaft Radius: 0.004000000189989805\n      Shape: Arrow\n      Topic:" in rviz_text

    assert "Name: Candidate Grasps" in rviz_text and "Value: true" in rviz_text.split("Name: Candidate Grasps", 1)[1].split("Name: Selected Grasp Pose", 1)[0]
    assert "Name: Selected Grasp Pose" in rviz_text and "Value: true" in rviz_text.split("Name: Selected Grasp Pose", 1)[1].split("Name: Plan Waypoints", 1)[0]
    assert "Name: Plan Waypoints" in rviz_text and "Value: true" in rviz_text.split("Name: Plan Waypoints", 1)[1].split("Enabled: true", 1)[0]
    assert "Name: Plan Markers" in rviz_text and "Value: true" in rviz_text.split("Name: Plan Markers", 1)[1].split("Name: Candidate Grasps", 1)[0]
    assert "/grasp_pipeline/rviz/candidate_validation_markers" in rviz_text


def test_snapshot_script_tracks_pose_visualization_topics():
    script_text = (PROJECT_ROOT / "scripts" / "show_last_distributed_snapshot.sh").read_text(
        encoding="utf-8"
    )

    assert "/vision_worker/rviz/candidate_grasps" in script_text
    assert "/vision_worker/rviz/selected_grasp" in script_text
    assert "/vision_worker/rviz/plan_waypoints" in script_text
    assert "/vision_worker/rviz/candidate_markers" in script_text
    assert "/vision_worker/rviz/selected_grasp_markers" in script_text
    assert "/vision_worker/rviz/plan_markers" in script_text
    assert "/grasp_pipeline/rviz/candidate_validation_markers" in script_text
