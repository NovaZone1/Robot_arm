import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_grasp_ros2.distributed_utils import build_runtime_config
from src.grasping.models import GraspExecutionConfig
from src.run_grasp_pipeline_ros2 import build_parser


ROBOTARM_EXP_ROOT = Path("/home/justahorse/Document/robotarm_exp")
CURRENT_DEFAULT_HAND_EYE_CONFIG = PROJECT_ROOT / "config" / "hand_eye" / "verify_config.yaml"


def test_build_parser_keeps_legacy_workspace_z_default():
    parser = build_parser()

    args = parser.parse_args(["cup"])

    assert tuple(args.workspace_z) == (0.0, 0.60)


def test_build_parser_defaults_match_robotarm_exp_runtime():
    parser = build_parser()

    args = parser.parse_args(["cup"])

    assert args.hand_eye_config == str(CURRENT_DEFAULT_HAND_EYE_CONFIG)
    assert args.npoint_tool_offset_file == ""
    assert args.online_bias_file == ""
    assert args.pointcloud_backend == "sdk"
    assert args.speed == 40
    assert args.gripper_open_mm == 70.0
    assert args.grasp_close_effort_nm == 0.6
    assert args.gripper_length_m == 0.105
    assert args.tool_contact_offset_scale == 1.0
    assert args.max_reachable_rotation_delta_deg == 180.0
    assert args.min_gripper_table_clearance_m == 0.03
    assert tuple(args.home_pose) == (57.0, 0.0, 215.0, 0.0, 85.0, 0.0)
    assert tuple(args.handoff_pose) == (200.0, 20.0, 300.0, 10.0, 120.0, 0.0)
    assert tuple(args.observe_pose) == (30.0, 0.0, 400.0, 0.0, 120.0, 0.0)


def test_grasp_execution_config_defaults_match_runtime_defaults():
    config = GraspExecutionConfig(hand_eye_config_path="dummy.yaml")

    assert config.robot_speed_percent == 40
    assert config.home_pose_mm_deg == (57.0, 0.0, 215.0, 0.0, 85.0, 0.0)
    assert config.handoff_pose_mm_deg == (200.0, 20.0, 300.0, 10.0, 120.0, 0.0)
    assert config.observe_pose_mm_deg == (30.0, 0.0, 400.0, 0.0, 120.0, 0.0)
    assert config.precenter_before_grasp is False
    assert config.max_reachable_rotation_delta_deg == 180.0
    assert config.gripper_length_m == 0.105
    assert config.grasp_close_effort_nm == 0.6
    assert config.min_gripper_table_clearance_m == 0.03


def test_distributed_pipeline_defaults_use_sdk_backend():
    yaml_text = (
        PROJECT_ROOT / "config" / "distributed" / "pipeline_orchestrator.params.yaml"
    ).read_text(encoding="utf-8")
    source_text = (
        PROJECT_ROOT / "robot_grasp_ros2" / "pipeline_orchestrator_node.py"
    ).read_text(encoding="utf-8")

    assert 'pointcloud_backend: "sdk"' in yaml_text
    assert 'self.declare_parameter("pointcloud_backend", "sdk")' in source_text
    assert '"pointcloud_backend": str(self.get_parameter("pointcloud_backend").value or "sdk")' in source_text


def test_distributed_pipeline_defaults_to_photo_card_target_then_base_scan():
    yaml_text = (
        PROJECT_ROOT / "config" / "distributed" / "pipeline_orchestrator.params.yaml"
    ).read_text(encoding="utf-8")
    dashboard_source = (
        PROJECT_ROOT / "scripts" / "run_grasp_dashboard.py"
    ).read_text(encoding="utf-8")

    red_flag_script = (
        PROJECT_ROOT / "scripts" / "wait_for_red_flag_start.sh"
    ).read_text(encoding="utf-8")

    assert "auto_target_from_card: true" in yaml_text
    assert "target_card_min_confidence: 0.53" in yaml_text
    assert "target_card_min_margin: 0.08" in yaml_text
    assert "target_card_search_roi_norm: [0.35, 0.01, 0.88, 0.50]" in yaml_text
    assert "target_card_capture_frames: 3" in yaml_text
    assert "target_card_consensus_frames: 2" in yaml_text
    assert "observation_speed: 25" in yaml_text
    assert "home_speed: 25" in yaml_text
    assert "speed_percent: 25.0" in red_flag_script
    assert "speed_percent: 5.0" not in red_flag_script
    assert "base_grasp_scan_enabled: true" in yaml_text
    assert 'id="autoTargetCardInput" type="checkbox" checked' in dashboard_source


def test_distributed_competition_defaults_use_center_horizontal_grasp():
    pipeline_yaml = (
        PROJECT_ROOT / "config" / "distributed" / "pipeline_orchestrator.params.yaml"
    ).read_text(encoding="utf-8")
    executor_yaml = (
        PROJECT_ROOT / "config" / "distributed" / "robot_executor.params.yaml"
    ).read_text(encoding="utf-8")
    dashboard_source = (PROJECT_ROOT / "scripts" / "run_grasp_dashboard.py").read_text(encoding="utf-8")
    executor_source = (PROJECT_ROOT / "robot_grasp_ros2" / "robot_executor_node.py").read_text(
        encoding="utf-8"
    )

    assert 'ARTIFACT_ROOT = BUNDLE_ROOT / "log" / "distributed_runs"' in dashboard_source
    assert "stack_ready = all(components.values())" in dashboard_source
    assert '"timeout_s": 12.0' not in executor_source
    assert "timeout_s=self._plan_pose_timeout_s()" in executor_source
    assert "use_object_center_contact: true" in pipeline_yaml
    assert "object_center_contact_max_offset_m: 0.08" in pipeline_yaml
    assert 'execution_strategy: "center_horizontal"' in executor_yaml
    assert "top_down_rpy_deg: [180.0, 85.0, 90.0]" in executor_yaml
    assert "center_horizontal_follow_target_azimuth: true" in executor_yaml
    assert "center_horizontal_reference_azimuth_deg: -90.0" in executor_yaml
    assert "top_down_lift_to_safe_z: false" in executor_yaml
    assert "top_down_vertical_step_mm: 80.0" in executor_yaml
    assert "top_down_max_speed_percent: 100.0" in executor_yaml
    assert "default_speed_percent: 25.0" in executor_yaml
    assert "home_speed_percent: 25.0" in executor_yaml
    assert "placement_speed_percent: 25.0" in executor_yaml
    assert "placement_final_speed_percent: 5.0" in executor_yaml
    assert "safe_top_down_final_speed_percent: 5.0" in executor_yaml
    assert "speed: 25" in pipeline_yaml
    assert "home_speed: 25" in pipeline_yaml
    assert 'id="centerContactInput" type="checkbox" checked' in dashboard_source
    assert 'id="speedRangeInput" type="range" min="1" max="100" step="1" value="25"' in dashboard_source
    assert (
        'data-prompt="red block" data-center-contact="true" '
        'data-execution-strategy="safe_top_down"'
    ) in dashboard_source
    assert (
        'data-prompt="yellow block" data-center-contact="true" '
        'data-execution-strategy="safe_top_down"'
    ) in dashboard_source
    assert (
        'data-prompt="blue block" data-center-contact="true" '
        'data-execution-strategy="safe_top_down"'
    ) in dashboard_source
    assert 'id="executionStrategyInput"' in dashboard_source
    assert 'execution_strategy: document.getElementById("executionStrategyInput").value' in dashboard_source


def test_build_runtime_config_applies_npoint_tool_offset(tmp_path):
    offset_file = tmp_path / "npoint_tool_offset.json"
    offset_file.write_text(
        json.dumps({"tool_contact_offset_tool_mm": [-13.25, 1.75, 157.5]}),
        encoding="utf-8",
    )

    config, _summary = build_runtime_config(
        {
            "prompt": "cup",
            "hand_eye_config": str(ROBOTARM_EXP_ROOT / "hand_eye_calibrate" / "verify_config.yaml"),
            "apply_npoint_tool_offset": True,
            "npoint_tool_offset_file": str(offset_file),
        }
    )

    assert config.tool_contact_offset_tool_m == (-0.01325, 0.00175, 0.1575)


def test_build_runtime_config_scales_npoint_tool_offset(tmp_path):
    offset_file = tmp_path / "npoint_tool_offset.json"
    offset_file.write_text(
        json.dumps({"tool_contact_offset_tool_mm": [-10.0, 2.0, 100.0]}),
        encoding="utf-8",
    )

    config, summary = build_runtime_config(
        {
            "prompt": "cup",
            "hand_eye_config": str(ROBOTARM_EXP_ROOT / "hand_eye_calibrate" / "verify_config.yaml"),
            "apply_npoint_tool_offset": True,
            "npoint_tool_offset_file": str(offset_file),
            "tool_contact_offset_scale": 0.5,
        }
    )

    assert config.tool_contact_offset_tool_m == (-0.005, 0.001, 0.05)
    assert summary["tool_contact_offset_scale"] == 0.5


def test_distributed_pipeline_defaults_keep_npoint_tool_offset_opt_in():
    yaml_text = (
        PROJECT_ROOT / "config" / "distributed" / "pipeline_orchestrator.params.yaml"
    ).read_text(encoding="utf-8")
    source_text = (
        PROJECT_ROOT / "robot_grasp_ros2" / "pipeline_orchestrator_node.py"
    ).read_text(encoding="utf-8")

    assert "apply_npoint_tool_offset: false" in yaml_text
    assert 'npoint_tool_offset_file: ""' in yaml_text
    assert "tool_contact_offset_scale: 0.5" in yaml_text
    assert "pregrasp_offset_m: 0.08" in yaml_text
    assert "descend_offset_m: 0.015" in yaml_text
    assert "retreat_offset_m: 0.10" in yaml_text
    assert 'self.declare_parameter("apply_npoint_tool_offset", False)' in source_text
    assert 'self.declare_parameter("npoint_tool_offset_file", "")' in source_text
    assert 'self.declare_parameter("tool_contact_offset_scale", 1.0)' in source_text
    assert 'self.declare_parameter("pregrasp_offset_m", 0.0)' in source_text
    assert 'self.declare_parameter("descend_offset_m", 0.0)' in source_text
    assert 'self.declare_parameter("retreat_offset_m", 0.0)' in source_text
    assert '"apply_npoint_tool_offset": bool(self.get_parameter("apply_npoint_tool_offset").value)' in source_text
    assert '"npoint_tool_offset_file": str(self.get_parameter("npoint_tool_offset_file").value or "")' in source_text
    assert '"tool_contact_offset_scale": float(self.get_parameter("tool_contact_offset_scale").value)' in source_text
