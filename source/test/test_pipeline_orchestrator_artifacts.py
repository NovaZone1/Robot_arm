import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

import robot_grasp_ros2.pipeline_orchestrator_node as orchestrator_module
from robot_grasp_ros2.pipeline_orchestrator_node import PipelineOrchestratorNode
from src.grasping.models import GraspCandidate, GraspPlan
from src.perception.item_catalog import ItemCatalog, LabelDetection


def test_target_card_selection_requires_clear_catalog_winner():
    detections = [
        LabelDetection("red_block", 0.82, (10, 20, 80, 90), "template"),
        LabelDetection("blue_block", 0.61, (120, 20, 80, 90), "template"),
    ]

    selected = PipelineOrchestratorNode._select_target_card_detection(
        detections,
        minimum_confidence=0.55,
        minimum_margin=0.08,
    )

    assert selected.item_id == "red_block"


@pytest.mark.parametrize(
    ("detections", "message"),
    [
        ([], "no catalog image"),
        ([LabelDetection("red_block", 0.40, (0, 0, 10, 10))], "too low"),
        (
            [
                LabelDetection("red_block", 0.75, (0, 0, 10, 10)),
                LabelDetection("blue_block", 0.71, (20, 0, 10, 10)),
            ],
            "ambiguous",
        ),
    ],
)
def test_target_card_selection_fails_closed(detections, message):
    with pytest.raises(RuntimeError, match=message):
        PipelineOrchestratorNode._select_target_card_detection(
            detections,
            minimum_confidence=0.55,
            minimum_margin=0.08,
        )


def test_target_card_consensus_requires_repeated_clear_winner():
    frames = [
        (
            LabelDetection("orange_bottle", 0.81, (10, 20, 30, 80), "template"),
        ),
        (
            LabelDetection("orange_bottle", 0.76, (11, 20, 30, 80), "template"),
        ),
        (
            LabelDetection("blue_block", 0.72, (40, 20, 70, 70), "template"),
            LabelDetection("red_block", 0.70, (120, 20, 70, 70), "template"),
        ),
    ]

    selected, diagnostics = PipelineOrchestratorNode._select_target_card_consensus(
        frames,
        minimum_confidence=0.55,
        minimum_margin=0.08,
        minimum_votes=2,
    )

    assert selected.item_id == "orange_bottle"
    assert diagnostics["winning_votes"] == 2
    assert len(diagnostics["rejected_frames"]) == 1


def test_target_card_consensus_rejects_single_frame_false_positive():
    frames = [
        (
            LabelDetection("blue_block", 0.79, (10, 20, 80, 80), "template"),
        ),
        (),
        (),
    ]

    with pytest.raises(RuntimeError, match="consensus failed"):
        PipelineOrchestratorNode._select_target_card_consensus(
            frames,
            minimum_confidence=0.55,
            minimum_margin=0.08,
            minimum_votes=2,
        )


def test_successful_grasp_moves_to_observation_without_enabling_auto_place(
    monkeypatch,
):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "place_after_grasp": False,
        "move_to_placement_observation_after_grasp": True,
        "placement_observe_pose": [0.0, 35.5, 491.1, 180.0, 67.77, -89.97],
        "speed": 5,
        "observation_speed": 10,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "grasp_plan_to_msg",
        lambda _plan: orchestrator_module.ExecuteGraspPlan.Request().plan,
    )
    node._execute_plan_client = SimpleNamespace()
    node._call_client = lambda *_args, **_kwargs: SimpleNamespace(
        success=True,
        message="ok",
        execution_json=json.dumps({"status": "ok", "release_performed": False}),
    )
    named_pose_calls = []

    def execute_named_pose(**kwargs):
        named_pose_calls.append(kwargs)
        return SimpleNamespace(
            actual_pose=SimpleNamespace(
                x_mm=-0.03,
                y_mm=35.42,
                z_mm=491.24,
                roll_deg=180.0,
                pitch_deg=67.73,
                yaw_deg=-89.95,
            )
        )

    node._execute_named_pose = execute_named_pose
    node._publish_status = lambda _message: None

    result = PipelineOrchestratorNode._execute_grasp_and_optional_place(
        node,
        run_id="grasp-test",
        plan=SimpleNamespace(),
        move_home_after=False,
        target_item_id="blue_block",
        hand_eye=np.eye(4),
    )

    assert len(named_pose_calls) == 1
    assert named_pose_calls[0]["name"] == "placement_observation"
    assert named_pose_calls[0]["open_gripper_first"] is False
    assert result["release_performed"] is False
    assert result["placement_observation_after_grasp"]["gripper_opened"] is False
    assert result["placement_observation_after_grasp"]["actual_pose_mm_deg"][
        "z_mm"
    ] == pytest.approx(491.24)


@pytest.mark.parametrize("target_item_id", ["green_bottle", "red_block"])
def test_scanned_catalog_target_advances_base_after_retreat_before_observation(
    monkeypatch,
    target_item_id,
):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "place_after_grasp": False,
        "move_to_placement_observation_after_grasp": True,
        "placement_observe_pose": [0.0, 35.5, 491.1, 180.0, 67.77, -89.97],
        "post_grasp_base_advance_m": 1.5,
        "post_grasp_base_advance_speed_mps": 0.16,
        "post_grasp_base_advance_timeout_s": 50.0,
        "speed": 5,
        "observation_speed": 10,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "grasp_plan_to_msg",
        lambda _plan: orchestrator_module.ExecuteGraspPlan.Request().plan,
    )
    events = []
    node._execute_plan_client = SimpleNamespace()

    def call_client(*_args, **_kwargs):
        events.append("grasp_retreat_complete")
        return SimpleNamespace(
            success=True,
            message="ok",
            execution_json=json.dumps(
                {"status": "ok", "release_performed": False}
            ),
        )

    node._call_client = call_client
    observation_started = threading.Event()

    def move_base(distance_m, *, timeout_s=None, speed_mps=None):
        events.append("base_advance")
        assert distance_m == pytest.approx(1.5)
        assert timeout_s == pytest.approx(50.0)
        assert speed_mps == pytest.approx(0.16)
        assert observation_started.wait(timeout=1.0)
        return {
            "success": True,
            "traveled_m": 1.49,
            "requested_distance_m": distance_m,
        }

    node._move_base_for_scan = move_base

    def execute_named_pose(**_kwargs):
        events.append("placement_observation")
        observation_started.set()
        return SimpleNamespace(
            actual_pose=SimpleNamespace(
                x_mm=0.0,
                y_mm=35.5,
                z_mm=491.1,
                roll_deg=180.0,
                pitch_deg=67.77,
                yaw_deg=-89.97,
            )
        )

    node._execute_named_pose = execute_named_pose
    node._publish_status = lambda _message: None

    result = PipelineOrchestratorNode._execute_grasp_and_optional_place(
        node,
        run_id="grasp-test",
        plan=SimpleNamespace(),
        move_home_after=False,
        target_item_id=target_item_id,
        hand_eye=np.eye(4),
        advance_base_after_grasp=True,
    )

    assert events == [
        "grasp_retreat_complete",
        "base_advance",
        "placement_observation",
    ]
    assert result["post_grasp_base_advance"]["traveled_m"] == pytest.approx(
        1.49
    )
    assert result["post_grasp_base_advance"]["phase"] == (
        "during_placement_observation_after_grasp_retreat"
    )


def test_post_place_advance_is_one_continuous_1_5m_move(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "post_place_base_advance_m": 1.5,
        "post_place_base_advance_speed_mps": 0.16,
        "post_place_base_advance_timeout_s": 50.0,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._publish_status = lambda _message: None
    calls = []

    def move_base(distance_m, *, timeout_s=None, speed_mps=None):
        calls.append((distance_m, timeout_s, speed_mps))
        return {
            "success": True,
            "requested_distance_m": distance_m,
            "traveled_m": 1.49,
        }

    node._move_base_for_scan = move_base

    result = PipelineOrchestratorNode._advance_base_after_place(
        node,
        run_id="place-test",
        item_id="blue_block",
    )

    assert calls == [(1.5, 50.0, 0.16)]
    assert result["phase"] == "after_place_release_and_retreat"
    assert result["traveled_m"] == pytest.approx(1.49)


def _retry_return_node(monkeypatch, parameters=None):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    values = {
        "base_multiview_speed_mps": 0.08,
        "base_multiview_move_timeout_s": 22.0,
        "target_card_base_search_speed_mps": 0.03,
        "target_card_base_search_timeout_s": 18.0,
        "observation_speed": 20.0,
        "observe_pose": [30.0, 0.0, 400.0, 0.0, 120.0, 0.0],
    }
    if parameters:
        values.update(parameters)
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=values[name]),
    )
    node._publish_status = lambda _message: None
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda _message: None,
        error=lambda _message: None,
        info=lambda _message: None,
    )
    node._execute_named_pose = lambda **_kwargs: None
    return node


def test_recognition_retry_reverses_full_scan_distance_at_scan_speed(monkeypatch):
    node = _retry_return_node(monkeypatch)
    calls = []

    def move_base(distance_m, *, timeout_s=None, speed_mps=None):
        calls.append((distance_m, timeout_s, speed_mps))
        return {
            "success": True,
            "requested_distance_m": distance_m,
            "traveled_m": abs(distance_m),
        }

    node._move_base_for_scan = move_base
    PipelineOrchestratorNode._return_to_observation_for_retry(
        node,
        run_id="grasp-test",
        reverse_m=1.489,
        reason="grasp_item_retry_2",
    )

    assert len(calls) == 1
    distance_m, timeout_s, speed_mps = calls[0]
    assert distance_m == pytest.approx(-1.489)
    assert speed_mps == pytest.approx(0.08)
    assert speed_mps != pytest.approx(0.03)
    assert timeout_s == pytest.approx(1.489 / 0.08 + 8.0)
    assert timeout_s > 18.0


def test_recognition_retry_keeps_reversing_until_scan_origin(monkeypatch):
    node = _retry_return_node(monkeypatch)
    remaining_steps = [0.50, 0.988]
    calls = []

    def move_base(distance_m, *, timeout_s=None, speed_mps=None):
        step = remaining_steps.pop(0)
        calls.append((distance_m, timeout_s, speed_mps))
        return {
            "success": True,
            "requested_distance_m": distance_m,
            "traveled_m": min(step, abs(distance_m)),
        }

    node._move_base_for_scan = move_base
    payload = PipelineOrchestratorNode._reverse_scan_travel(node, 1.488)

    assert payload["complete"] is True
    assert payload["traveled_m"] == pytest.approx(1.488)
    assert payload["remaining_m"] == pytest.approx(0.0)
    assert [call[0] for call in calls] == pytest.approx([-1.488, -0.988])
    assert all(call[2] == pytest.approx(0.08) for call in calls)


def test_post_place_home_and_base_advance_start_in_parallel(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "post_place_home_pose": [57.0, 0.0, 215.0, 0.0, 85.0, 0.0],
        "speed": 5.0,
        "home_speed": 25.0,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._publish_status = lambda _message: None
    node._request_base_scan_stop = lambda: None
    base_started = threading.Event()
    allow_base_finish = threading.Event()

    def advance_base(**_kwargs):
        base_started.set()
        assert allow_base_finish.wait(timeout=1.0)
        return {"success": True, "traveled_m": 1.49}

    def execute_home(**kwargs):
        assert kwargs["speed_percent"] == 25.0
        assert base_started.wait(timeout=1.0)
        allow_base_finish.set()
        return SimpleNamespace(
            actual_pose=SimpleNamespace(
                x_mm=57.0,
                y_mm=0.0,
                z_mm=215.0,
                roll_deg=0.0,
                pitch_deg=85.0,
                yaw_deg=0.0,
            )
        )

    node._advance_base_after_place = advance_base
    node._execute_named_pose = execute_home

    home, movement = (
        PipelineOrchestratorNode._return_home_and_advance_base_after_place(
            node,
            run_id="place-test",
            item_id="blue_block",
        )
    )

    assert home["success"] is True
    assert movement["traveled_m"] == pytest.approx(1.49)


def test_base_grasp_scan_stops_at_first_view_with_a_candidate(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda item_id: SimpleNamespace(item_id=item_id, kind="block")
    )
    parameters = {
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 1.20,
        "base_multiview_max_views": 10,
        "base_multiview_settle_s": 0.0,
        "base_grasp_bottle_center_norm": [0.598, 0.485],
        "base_grasp_block_center_norm": [0.606, 0.619],
        "base_grasp_center_tolerance_u_norm": 0.08,
        "base_grasp_center_tolerance_v_norm": 0.12,
        "base_target_fine_step_m": 0.07,
        "depth_fusion_frames": 8,
        "continuous_search_enabled": False,
        "continuous_search_stop_on_center": False,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._publish_status = lambda _message: None
    node._request_base_scan_stop = lambda: None
    node._move_base_for_scan = lambda _distance: pytest.fail(
        "base must not move after target is found in the first view"
    )
    capture = SimpleNamespace(
        scene_id="scene-start",
        camera_info=SimpleNamespace(
            width=640,
            height=480,
            k=np.asarray(
                [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
            ),
        ),
    )
    analyze = SimpleNamespace()
    candidate = SimpleNamespace(
        score=0.9,
        object_center_camera_m=(0.06784, 0.05712, 0.6),
    )
    cycle = {
        "capture_response": capture,
        "analyze_response": analyze,
        "candidate_pool": [candidate],
        "candidate": candidate,
    }
    captured = {
        "state_snapshot": {},
        "capture_response": capture,
        "detection_response": SimpleNamespace(
            found=True,
            center_u_norm=0.606,
            center_v_norm=0.619,
            confidence=0.95,
            backend="test",
        ),
    }
    node._capture_detect_target_2d = lambda **_kwargs: captured
    node._analyze_captured_cycle = lambda **_kwargs: cycle
    node._capture_debug_dict = lambda _capture: {"scene_id": _capture.scene_id}
    node._analyze_debug_dict = lambda _analyze: {"candidate_count": 1}

    selected, scan = PipelineOrchestratorNode._run_base_grasp_target_scan(
        node,
        run_id="grasp-test",
        prompt="blue block",
        target_item_id="blue_block",
        options={},
        hand_eye=np.eye(4),
    )

    assert selected["candidate"] is candidate
    assert scan["success"] is True
    assert scan["motion_command_sent"] is False
    assert scan["views"][0]["view_name"] == "start"
    assert scan["views"][0]["target_centered"] is True


def test_base_grasp_scan_fine_steps_until_target_is_centered(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda item_id: SimpleNamespace(item_id=item_id, kind="bottle")
    )
    parameters = {
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 1.20,
        "base_multiview_max_views": 3,
        "base_multiview_settle_s": 0.0,
        "base_grasp_bottle_center_norm": [0.598, 0.485],
        "base_grasp_block_center_norm": [0.606, 0.619],
        "base_grasp_center_tolerance_u_norm": 0.08,
        "base_grasp_center_tolerance_v_norm": 0.12,
        "base_target_fine_step_m": 0.07,
        "depth_fusion_frames": 8,
        "continuous_search_enabled": False,
        "continuous_search_stop_on_center": False,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._publish_status = lambda _message: None
    node._request_base_scan_stop = lambda: None
    movements = []

    def move(distance):
        movements.append(distance)
        return {"traveled_m": distance, "success": True}

    node._move_base_for_scan = move
    camera_info = SimpleNamespace(
        width=640,
        height=480,
        k=[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
    )
    centers = iter(((0.80, 0.485), (0.598, 0.485)))
    last_center = {"value": (0.80, 0.485)}

    def capture_detect(**kwargs):
        if kwargs.get("depth_fusion_frames") is None:
            last_center["value"] = next(centers)
        center_u, center_v = last_center["value"]
        return {
            "state_snapshot": {},
            "capture_response": SimpleNamespace(
                scene_id="scene", camera_info=camera_info
            ),
            "detection_response": SimpleNamespace(
                found=True,
                center_u_norm=center_u,
                center_v_norm=center_v,
                confidence=0.9,
                backend="test",
            ),
        }

    candidate = SimpleNamespace(
        score=0.9,
        object_center_camera_m=(0.06272, -0.0072, 0.6),
    )
    cycle = {
        "capture_response": SimpleNamespace(scene_id="scene", camera_info=camera_info),
        "analyze_response": SimpleNamespace(),
        "candidate_pool": [candidate],
        "candidate": candidate,
    }
    node._capture_detect_target_2d = capture_detect
    node._analyze_captured_cycle = lambda **_kwargs: cycle
    node._capture_debug_dict = lambda _capture: {"scene_id": "scene"}
    node._analyze_debug_dict = lambda _analyze: {"candidate_count": 1}

    _selected, scan = PipelineOrchestratorNode._run_base_grasp_target_scan(
        node,
        run_id="grasp-test",
        prompt="green bottle",
        target_item_id="green_bottle",
        options={},
        hand_eye=np.eye(4),
    )

    assert movements == [pytest.approx(0.07)]
    assert scan["success"] is True
    assert scan["views"][0]["target_centered"] is False
    assert scan["views"][1]["target_centered"] is True


def test_base_grasp_scan_rejects_scene_fallback_candidate(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda item_id: SimpleNamespace(item_id=item_id, kind="bottle")
    )
    parameters = {
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 0.0,
        "base_multiview_max_views": 1,
        "base_multiview_settle_s": 0.0,
        "base_grasp_bottle_center_norm": [0.598, 0.485],
        "base_grasp_block_center_norm": [0.606, 0.619],
        "base_grasp_center_tolerance_u_norm": 0.08,
        "base_grasp_center_tolerance_v_norm": 0.12,
        "base_target_fine_step_m": 0.07,
        "depth_fusion_frames": 8,
        "continuous_search_enabled": False,
        "continuous_search_stop_on_center": False,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._publish_status = lambda _message: None
    node._request_base_scan_stop = lambda: None
    node._move_base_for_scan = lambda _distance: pytest.fail("must not move")
    fallback = SimpleNamespace(score=0.9, object_center_camera_m=None)
    cycle = {
        "capture_response": SimpleNamespace(
            scene_id="scene",
            camera_info=SimpleNamespace(
                width=640,
                height=480,
                k=[600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
            ),
        ),
        "analyze_response": SimpleNamespace(),
        "candidate_pool": [fallback],
        "candidate": fallback,
    }
    captured = {
        "state_snapshot": {},
        "capture_response": cycle["capture_response"],
        "detection_response": SimpleNamespace(
            found=True,
            center_u_norm=0.598,
            center_v_norm=0.485,
            confidence=0.9,
            backend="test",
        ),
    }
    node._capture_detect_target_2d = lambda **_kwargs: captured
    node._analyze_captured_cycle = lambda **_kwargs: cycle
    node._capture_debug_dict = lambda _capture: {"scene_id": "scene"}
    node._analyze_debug_dict = lambda _analyze: {"candidate_count": 1}

    selected, scan = PipelineOrchestratorNode._run_base_grasp_target_scan(
        node,
        run_id="grasp-test",
        prompt="green bottle",
        target_item_id="green_bottle",
        options={},
        hand_eye=np.eye(4),
    )

    assert selected is cycle
    assert scan["success"] is False
    assert scan["views"][0]["target_centered"] is False


def _aligned_place_node(monkeypatch, tmp_path, *, odom=(1.0, 2.0, 0.1)):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._placement_scan_viz_dir = lambda: tmp_path
    node._base_odom_snapshot = lambda: odom
    parameters = {
        "cached_box_map_max_age_s": 600.0,
        "cached_box_map_position_tolerance_m": 0.025,
        "cached_box_map_yaw_tolerance_deg": 2.0,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    payload = {
        "success": True,
        "scan_mode": "base_single_pass",
        "created_at_unix_s": time.time(),
        "base_odom_origin": {"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.1},
        "fused_map": {"item_to_slot_index": {"blue_block": 0}},
        "target_alignment": {
            "success": True,
            "item_id": "blue_block",
            "slot_index": 0,
            "selected_confidence": 0.91,
        },
    }
    (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
    return node


def test_aligned_place_context_accepts_matching_fresh_scan(monkeypatch, tmp_path):
    node = _aligned_place_node(monkeypatch, tmp_path)

    payload, slot_index, confidence = (
        PipelineOrchestratorNode._validated_aligned_place_context(
            node, "blue_block"
        )
    )

    assert payload["target_alignment"]["item_id"] == "blue_block"
    assert slot_index == 0
    assert confidence == pytest.approx(0.91)


def test_aligned_place_context_rejects_base_motion_after_alignment(
    monkeypatch, tmp_path
):
    node = _aligned_place_node(monkeypatch, tmp_path, odom=(1.04, 2.0, 0.1))

    with pytest.raises(RuntimeError, match="moved after target alignment"):
        PipelineOrchestratorNode._validated_aligned_place_context(
            node, "blue_block"
        )


def test_aligned_place_context_rejects_different_target(monkeypatch, tmp_path):
    node = _aligned_place_node(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="not red_block"):
        PipelineOrchestratorNode._validated_aligned_place_context(
            node, "red_block"
        )


def test_aligned_place_context_accepts_centered_target_only_scan(
    monkeypatch, tmp_path
):
    node = _aligned_place_node(monkeypatch, tmp_path)
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    payload["scan_mode"] = "base_target_single_pass"
    payload.pop("fused_map")
    (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    _payload, slot_index, confidence = (
        PipelineOrchestratorNode._validated_aligned_place_context(
            node, "blue_block"
        )
    )

    assert slot_index == 0
    assert confidence == pytest.approx(0.91)


def test_target_box_scan_accepts_centered_label_without_six_label_row(
    monkeypatch, tmp_path
):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._run_lock = threading.Lock()
    node._scan_active = False
    node._run_thread = None
    node._stop_requested = False
    node._result_pub = SimpleNamespace(publish=lambda _message: None)
    parameters = {
        "base_target_alignment_enabled": True,
        "target_item_id": "blue_block",
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 1.2,
        "base_multiview_max_views": 10,
        "base_target_center_tolerance_norm": 0.18,
        "base_target_fine_step_m": 0.03,
        "label_match_threshold": 0.42,
        "base_multiview_settle_s": 0.0,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._placement_scan_viz_dir = lambda: tmp_path
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda _item_id: SimpleNamespace(item_id="blue_block")
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._build_runtime = lambda: ({}, SimpleNamespace(), np.eye(4), SimpleNamespace())
    node._read_robot_state_snapshot = lambda **_kwargs: {
        "base_to_camera": np.eye(4),
        "current_pose": SimpleNamespace(
            x_mm=0.0, y_mm=0.0, z_mm=491.0,
            roll_deg=180.0, pitch_deg=68.0, yaw_deg=-90.0,
        ),
    }
    node._pose_debug_dict = lambda _pose: {"z_mm": _pose.z_mm}
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    node._capture_placement_scan_view = lambda **_kwargs: {
        "view_name": "start",
        "offset_from_start_m": 0.0,
        "capture": {"color_width": 640},
        # The old matcher reports success=False without six labels. The new
        # target-only alignment must still accept this centered target label.
        "label_match": {
            "success": False,
            "matched_item_id": "blue_block",
            "confidence": 0.91,
            "bbox_xywh": [300, 100, 40, 60],
        },
        "images": {},
    }
    node._request_base_scan_stop = lambda: None
    response = SimpleNamespace(success=False, message="")

    result = PipelineOrchestratorNode._handle_scan_and_align_placement_target_service(
        node, SimpleNamespace(), response
    )

    assert result.success is True
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["scan_mode"] == "base_target_single_pass"
    assert payload["target_alignment"]["success"] is True
    assert payload["movements"] == []


def test_target_box_scan_reverses_after_overshoot(monkeypatch, tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._run_lock = threading.Lock()
    node._scan_active = False
    node._run_thread = None
    node._stop_requested = False
    node._result_pub = SimpleNamespace(publish=lambda _message: None)
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    parameters = {
        "base_target_alignment_enabled": True,
        "target_item_id": "blue_block",
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 1.2,
        "base_multiview_max_views": 6,
        "base_target_center_tolerance_norm": 0.12,
        "base_target_fine_step_m": 0.07,
        "label_match_threshold": 0.42,
        "base_multiview_settle_s": 0.0,
        "placement_scan_max_retries": 0,
        "continuous_search_enabled": False,
        "continuous_search_stop_on_center": False,
        "observation_speed": 25,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    node._placement_scan_viz_dir = lambda: tmp_path
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda _item_id: SimpleNamespace(item_id="blue_block")
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._build_runtime = lambda: ({}, SimpleNamespace(), np.eye(4), SimpleNamespace())
    node._read_robot_state_snapshot = lambda **_kwargs: {
        "base_to_camera": np.eye(4),
        "current_pose": SimpleNamespace(
            x_mm=0.0, y_mm=0.0, z_mm=491.0,
            roll_deg=180.0, pitch_deg=68.0, yaw_deg=-90.0,
        ),
    }
    node._pose_debug_dict = lambda _pose: {"z_mm": _pose.z_mm}
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    node._publish_status = lambda _text: None
    labels = [
        {"matched_item_id": "blue_block", "confidence": 0.9, "bbox_xywh": [200, 100, 40, 60]},
        {"matched_item_id": "blue_block", "confidence": 0.9, "bbox_xywh": [80, 100, 40, 60]},
        {"matched_item_id": "blue_block", "confidence": 0.9, "bbox_xywh": [300, 100, 40, 60]},
    ]
    call_index = {"value": 0}

    def capture_view(**kwargs):
        label = dict(labels[min(call_index["value"], len(labels) - 1)])
        call_index["value"] += 1
        return {
            "view_name": kwargs.get("view_name"),
            "offset_from_start_m": kwargs.get("offset_from_start_m"),
            "capture": {"color_width": 640},
            "label_match": label,
            "images": {},
        }

    moves: list[float] = []

    def move(distance_m, **_kwargs):
        moves.append(float(distance_m))
        return {
            "success": True,
            "message": "ok",
            "requested_distance_m": float(distance_m),
            "traveled_m": float(distance_m),
            "lateral_error_m": 0.0,
            "yaw_error_deg": 0.0,
        }

    node._capture_placement_scan_view = capture_view
    node._move_base_for_scan = move
    node._request_base_scan_stop = lambda: None
    response = SimpleNamespace(success=False, message="")

    result = PipelineOrchestratorNode._handle_scan_and_align_placement_target_service(
        node, SimpleNamespace(), response
    )

    assert result.success is True
    assert any(distance < 0.0 for distance in moves)
    assert moves[-1] < 0.0 or any(distance < 0.0 for distance in moves[:-1])


def test_target_box_scan_stops_on_taught_center_not_image_center(
    monkeypatch, tmp_path
):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._run_lock = threading.Lock()
    node._scan_active = False
    node._run_thread = None
    node._stop_requested = False
    node._result_pub = SimpleNamespace(publish=lambda _message: None)
    parameters = {
        "base_target_alignment_enabled": True,
        "target_item_id": "orange_bottle",
        "base_multiview_offset_m": 0.15,
        "base_multiview_max_travel_m": 1.2,
        "base_multiview_max_views": 6,
        "base_target_center_tolerance_norm": 0.08,
        "base_target_fine_step_m": 0.07,
        "label_match_threshold": 0.42,
        "base_multiview_settle_s": 0.0,
        "placement_scan_max_retries": 0,
        "continuous_search_enabled": False,
        "continuous_search_stop_on_center": False,
        "observation_speed": 25,
    }
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=parameters[name]),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_mapping_for_item",
        lambda _item_id: SimpleNamespace(
            align_u_px=395.0,
            align_v_px=117.5,
            alignment_u_norm=lambda width: 395.0 / float(width),
        ),
    )
    node._placement_scan_viz_dir = lambda: tmp_path
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda _item_id: SimpleNamespace(item_id="orange_bottle")
    )
    node._base_odom_snapshot = lambda: (0.0, 0.0, 0.0)
    node._build_runtime = lambda: ({}, SimpleNamespace(), np.eye(4), SimpleNamespace())
    node._read_robot_state_snapshot = lambda **_kwargs: {
        "base_to_camera": np.eye(4),
        "current_pose": SimpleNamespace(
            x_mm=0.0, y_mm=0.0, z_mm=491.0,
            roll_deg=180.0, pitch_deg=68.0, yaw_deg=-90.0,
        ),
    }
    node._pose_debug_dict = lambda _pose: {"z_mm": _pose.z_mm}
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    node._publish_status = lambda _text: None
    labels = [
        # Image-center box: old stop condition, but not the taught arm-facing view.
        {"matched_item_id": "orange_bottle", "confidence": 0.7, "bbox_xywh": [300, 100, 40, 60]},
        # Taught center sample (u=395).
        {"matched_item_id": "orange_bottle", "confidence": 0.7, "bbox_xywh": [376, 60, 38, 115]},
    ]
    call_index = {"value": 0}

    def capture_view(**kwargs):
        label = dict(labels[min(call_index["value"], len(labels) - 1)])
        call_index["value"] += 1
        return {
            "view_name": kwargs.get("view_name"),
            "offset_from_start_m": kwargs.get("offset_from_start_m"),
            "capture": {"color_width": 640},
            "label_match": label,
            "images": {},
        }

    moves: list[float] = []

    def move(distance_m, **_kwargs):
        moves.append(float(distance_m))
        return {
            "success": True,
            "message": "ok",
            "requested_distance_m": float(distance_m),
            "traveled_m": float(distance_m),
            "lateral_error_m": 0.0,
            "yaw_error_deg": 0.0,
        }

    node._capture_placement_scan_view = capture_view
    node._move_base_for_scan = move
    node._request_base_scan_stop = lambda: None
    response = SimpleNamespace(success=False, message="")

    result = PipelineOrchestratorNode._handle_scan_and_align_placement_target_service(
        node, SimpleNamespace(), response
    )

    assert result.success is True
    assert moves
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["target_alignment"]["align_source"] == "taught_center_sample"
    assert payload["target_alignment"]["align_u_px"] == pytest.approx(395.0)
    assert payload["target_alignment"]["selected_view_name"] != "start"


def test_remember_target_card_exclusion_only_covers_the_printed_photo():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    PipelineOrchestratorNode._remember_target_card_exclusion(
        node,
        {
            "success": True,
            "bbox_xywh": [276, 102, 38, 110],
            "image_width": 640,
            "image_height": 480,
        },
    )
    exclude = node._grasp_exclude_roi_norm
    assert exclude is not None
    assert node._grasp_search_roi_norm is None
    # Printed photo is ignored; the calibrated bottle center remains searchable.
    assert PipelineOrchestratorNode._norm_point_in_roi(0.45, 0.30, exclude)
    assert not PipelineOrchestratorNode._norm_point_in_roi(0.598, 0.485, exclude)


def test_write_run_artifacts_writes_execution_trace_file(tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    artifact_root = tmp_path / "distributed_runs"
    run_id = "run-456"

    node._artifact_root_dir = lambda: artifact_root
    node._run_artifact_dir = lambda incoming_run_id: artifact_root / incoming_run_id

    result_payload = {
        "status": "ok",
        "prompt": "cup",
        "scene_id": "scene-1",
        "execution": {
            "status": "ok",
            "execution_trace": [
                {
                    "step_name": "grasp",
                    "command_type": "move_pose",
                    "success": True,
                }
            ],
        },
    }

    artifact_dir = PipelineOrchestratorNode._write_run_artifacts(
        node,
        run_id=run_id,
        request_payload={"run_id": run_id},
        cycle_records=[],
        result_payload=result_payload,
    )

    trace_payload = json.loads((artifact_dir / "execution_trace.json").read_text(encoding="utf-8"))
    final_payload = json.loads((artifact_dir / "final_result.json").read_text(encoding="utf-8"))

    assert trace_payload["run_id"] == run_id
    assert trace_payload["execution_trace"][0]["step_name"] == "grasp"
    assert "execution_trace" not in final_payload["execution"]


def test_write_run_artifacts_writes_candidate_validation_file(tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    artifact_root = tmp_path / "distributed_runs"
    run_id = "run-789"

    node._artifact_root_dir = lambda: artifact_root
    node._run_artifact_dir = lambda incoming_run_id: artifact_root / incoming_run_id

    result_payload = {
        "status": "ok",
        "prompt": "cup",
        "scene_id": "scene-2",
        "candidate_validation": [
            {
                "candidate_index": 0,
                "selection_result": "rejected_by_robot_validation",
                "robot_validation_stage": "grasp",
                "ik_error_type": "timeout",
            },
            {
                "candidate_index": 1,
                "selection_result": "selected_for_execution",
                "robot_validation_result": "accepted",
            },
        ],
    }

    artifact_dir = PipelineOrchestratorNode._write_run_artifacts(
        node,
        run_id=run_id,
        request_payload={"run_id": run_id},
        cycle_records=[],
        result_payload=result_payload,
    )

    validation_payload = json.loads((artifact_dir / "candidate_validation.json").read_text(encoding="utf-8"))
    final_payload = json.loads((artifact_dir / "final_result.json").read_text(encoding="utf-8"))

    assert validation_payload["run_id"] == run_id
    assert validation_payload["candidate_validation"][0]["robot_validation_stage"] == "grasp"
    assert validation_payload["candidate_validation"][1]["selection_result"] == "selected_for_execution"
    assert "candidate_validation" not in final_payload


def test_retarget_plan_uses_object_center_and_preserves_planner_contact_offset(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
        "table_z_m": 0.161,
        "min_gripper_table_clearance_m": 0.03,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())

    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.10, 0.20, 0.30),
        rotation_camera=np.eye(3),
        object_center_camera_m=(0.14, 0.24, 0.34),
        center_offset_m=0.069,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(0.1, 0.2, 0.4),
        grasp_base_m=(0.1, 0.2, 0.35),
        retreat_base_m=(0.1, 0.2, 0.45),
        within_workspace=False,
        workspace_violations=["legacy grasp orientation below table"],
        target_contact_point_base_m=(0.10, 0.20, 0.35),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
    )

    assert np.allclose(
        retargeted.target_contact_point_base_m,
        (0.1235294118, 0.2117647059, 0.35),
    )
    assert retargeted.within_workspace is True
    assert retargeted.workspace_violations == []


def test_retarget_plan_rejects_unreliable_object_center(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.0, 0.0, 0.45),
        rotation_camera=np.eye(3),
        object_center_camera_m=(0.20, 0.0, 0.62),
        center_offset_m=0.17,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.0, 0.4, 0.2),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(0.0, 0.4, 0.3),
        grasp_base_m=(0.0, 0.4, 0.25),
        retreat_base_m=(0.0, 0.4, 0.35),
        within_workspace=True,
        workspace_violations=[],
        target_contact_point_base_m=(0.0, 0.4, 0.22),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
    )

    assert retargeted.within_workspace is False
    assert "object center offset" in retargeted.workspace_violations[0]
    assert "exceeds" in retargeted.workspace_violations[0]


def test_retarget_color_block_uses_known_table_relative_center_height(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
        "table_z_m": 0.161,
        "min_gripper_table_clearance_m": 0.03,
        "color_block_center_height_m": 0.045,
        "manual_target_bias_z_mm": 0.0,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(-0.06, -0.06, 0.536),
        rotation_camera=np.eye(3),
        object_center_camera_m=(-0.046, -0.016, 0.532),
        center_offset_m=0.044,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(-0.1, 0.4, 0.2),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(-0.1, 0.4, 0.3),
        grasp_base_m=(-0.1, 0.4, 0.25),
        retreat_base_m=(-0.1, 0.4, 0.35),
        within_workspace=False,
        workspace_violations=["depth drift put contact below table clearance"],
        target_contact_point_base_m=(-0.1, 0.4, 0.17),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
        prompt="red block",
    )

    assert retargeted.target_contact_point_base_m[2] == pytest.approx(0.206)
    assert retargeted.within_workspace is True
    assert retargeted.workspace_violations == []


def test_scan_placement_captures_without_motion_or_gripper_commands(monkeypatch, tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    node._run_lock = threading.Lock()
    node._scan_active = False
    node._run_thread = None
    node._pending_confirmation = None
    node._result_pub = SimpleNamespace(publish=lambda _message: None)
    node._publish_status = lambda _text: None
    node._build_runtime = lambda: ({}, SimpleNamespace(), np.eye(4), SimpleNamespace())
    pose = SimpleNamespace(
        x_mm=10.0,
        y_mm=20.0,
        z_mm=300.0,
        roll_deg=180.0,
        pitch_deg=60.0,
        yaw_deg=-90.0,
    )
    node._read_robot_state_snapshot = lambda **_kwargs: {
        "current_pose": pose,
        "base_to_camera": np.eye(4),
    }
    image = SimpleNamespace(width=640, height=480)
    camera_info = SimpleNamespace(width=640, height=480, k=[1.0] * 9)
    capture = SimpleNamespace(
        scene_id="placement-scene",
        camera_frame="camera",
        color_image=image,
        depth_image=image,
        camera_info=camera_info,
    )
    node._capture_scene_once = lambda **_kwargs: capture
    item = SimpleNamespace(item_id="yellow_block")
    node._item_catalog = lambda: SimpleNamespace(
        resolve=lambda _value: None,
        items={"yellow_block": item},
    )
    node._match_box_label_once = lambda **_kwargs: {
        "complete": False,
        "message": "complete six-label row not verified",
        "detected_label_count": 6,
        "slot_index": 2,
        "diagnostics": {"detections": []},
    }
    node._placement_scan_viz_dir = lambda: tmp_path
    node.get_parameter = lambda _name: SimpleNamespace(value="")
    monkeypatch.setattr(
        orchestrator_module,
        "color_msg_to_bgr",
        lambda _message: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "depth_msg_to_meters",
        lambda _message: np.ones((480, 640), dtype=np.float32),
    )
    monkeypatch.setattr(orchestrator_module.cv2, "imwrite", lambda *_args, **_kwargs: True)
    response = SimpleNamespace(success=False, message="")

    result = PipelineOrchestratorNode._handle_scan_placement_service(
        node,
        SimpleNamespace(),
        response,
    )

    assert result.success is True
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "six-label" in payload["validation_message"]
    assert payload["motion_command_sent"] is False
    assert payload["gripper_command_sent"] is False
    assert payload["robot_pose_mm_deg"]["z_mm"] == 300.0


def test_multiview_fusion_converts_scan_offsets_into_start_base_frame():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    catalog = ItemCatalog.load(
        Path(__file__).resolve().parents[1] / "config" / "item_catalog.yaml"
    )
    node._item_catalog = lambda: catalog
    node.get_parameter = lambda name: SimpleNamespace(
        value={"table_z_m": 0.161}[name]
    )
    order = (
        "dark_bottle",
        "green_bottle",
        "orange_bottle",
        "blue_block",
        "red_block",
        "yellow_block",
    )
    world_points = {
        item_id: (-0.45 + (index * 0.18), 0.75, 0.30)
        for index, item_id in enumerate(order)
    }
    views = []
    for view_name, offset, visible in (
        ("center", 0.0, order[1:5]),
        ("forward", 0.28, order[3:]),
        ("backward", -0.28, order[:3]),
    ):
        observations = []
        for item_id in visible:
            point = world_points[item_id]
            observations.append(
                {
                    "item_id": item_id,
                    "confidence": 0.9,
                    "method": "test",
                    "point_base_m": [point[0] - offset, point[1], point[2]],
                }
            )
        views.append(
            {
                "view_name": view_name,
                "offset_from_start_m": offset,
                "label_match": {
                    "diagnostics": {
                        "partial_label_observations_base_m": observations,
                    }
                },
            }
        )
    # A high-confidence image match can still carry invalid depth when the
    # held object occludes the RGB-D view. It must not become the fusion anchor.
    views[1]["label_match"]["diagnostics"][
        "partial_label_observations_base_m"
    ].append(
        {
            "item_id": "blue_block",
            "confidence": 0.99,
            "method": "test_bad_depth",
            "point_base_m": [0.09, 0.75, 0.08],
        }
    )

    fused = PipelineOrchestratorNode._fuse_multiview_box_map(
        node,
        views=views,
        item_id="red_block",
        base_to_camera_at_start=np.eye(4),
    )

    assert fused["detected_item_ids_left_to_right"] == list(order)
    assert fused["item_to_slot_index"]["red_block"] == 4
    assert fused["adjacent_pitch_mm"] == pytest.approx([180.0] * 5)
    assert len(fused["rejected_observations"]) == 1
    assert (
        fused["rejected_observations"][0]["item_id"]
        == "blue_block"
    )


def test_single_pass_label_fusion_recovers_order_from_overlapping_views():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    order = (
        "green_bottle",
        "dark_bottle",
        "orange_bottle",
        "blue_block",
        "yellow_block",
        "red_block",
    )
    node._item_catalog = lambda: SimpleNamespace(
        items={item_id: object() for item_id in order}
    )
    views = []
    for index, visible in enumerate((order[:3], order[1:5], order[3:])):
        views.append(
            {
                "view_name": f"forward_{index:02d}",
                "offset_from_start_m": index * 0.15,
                "label_match": {
                    "diagnostics": {
                        "detections": [
                            {
                                "item_id": item_id,
                                "confidence": 0.85 + (0.01 * detection_index),
                                "bbox_xywh": [
                                    30 + (100 * detection_index),
                                    80,
                                    60,
                                    100,
                                ],
                                "method": "test",
                            }
                            for detection_index, item_id in enumerate(visible)
                        ]
                    }
                },
            }
        )

    fused = PipelineOrchestratorNode._fuse_single_pass_label_map(
        node,
        views=views,
        item_id="yellow_block",
    )

    assert fused["detected_item_ids_left_to_right"] == list(order)
    assert fused["item_to_slot_index"]["yellow_block"] == 4
    assert fused["transparent_depth_used"] is False
    assert "item_to_box_center_base_m" not in fused


def test_single_pass_label_fusion_rejects_disconnected_order():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    order = (
        "green_bottle",
        "dark_bottle",
        "orange_bottle",
        "blue_block",
        "yellow_block",
        "red_block",
    )
    node._item_catalog = lambda: SimpleNamespace(
        items={item_id: object() for item_id in order}
    )
    views = []
    for name, visible in (("left", order[:3]), ("right", order[3:])):
        views.append(
            {
                "view_name": name,
                "offset_from_start_m": 0.0,
                "label_match": {
                    "diagnostics": {
                        "detections": [
                            {
                                "item_id": item_id,
                                "confidence": 0.9,
                                "bbox_xywh": [20 + (100 * index), 50, 50, 80],
                            }
                            for index, item_id in enumerate(visible)
                        ]
                    }
                },
            }
        )

    with pytest.raises(RuntimeError, match="ambiguous"):
        PipelineOrchestratorNode._fuse_single_pass_label_map(
            node,
            views=views,
            item_id="yellow_block",
        )


def test_single_pass_label_fusion_outvotes_one_weak_reversed_relation():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    order = (
        "blue_block",
        "green_bottle",
        "yellow_block",
        "dark_bottle",
        "red_block",
        "orange_bottle",
    )
    node._item_catalog = lambda: SimpleNamespace(
        items={item_id: object() for item_id in order}
    )

    def view(name, visible, confidences):
        return {
            "view_name": name,
            "offset_from_start_m": 0.0,
            "label_match": {
                "diagnostics": {
                    "detections": [
                        {
                            "item_id": item_id,
                            "confidence": confidence,
                            "bbox_xywh": [20 + (90 * index), 50, 50, 80],
                        }
                        for index, (item_id, confidence) in enumerate(
                            zip(visible, confidences)
                        )
                    ]
                }
            },
        }

    views = [
        view(
            "weak_false_order",
            ("blue_block", "dark_bottle", "green_bottle"),
            (0.91, 0.44, 0.67),
        ),
        view(
            "middle_1",
            (
                "blue_block",
                "green_bottle",
                "yellow_block",
                "dark_bottle",
            ),
            (0.91, 0.67, 0.90, 0.82),
        ),
        view(
            "middle_2",
            (
                "green_bottle",
                "yellow_block",
                "dark_bottle",
                "red_block",
            ),
            (0.73, 0.90, 0.75, 0.91),
        ),
        view(
            "right",
            ("dark_bottle", "red_block", "orange_bottle"),
            (0.82, 0.91, 0.74),
        ),
    ]

    fused = PipelineOrchestratorNode._fuse_single_pass_label_map(
        node,
        views=views,
        item_id="yellow_block",
    )

    assert fused["detected_item_ids_left_to_right"] == list(order)
    assert fused["localization_source"] == "2d_weighted_label_order"


def test_single_pass_label_fusion_ignores_partial_boundary_detection():
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    order = (
        "blue_block",
        "green_bottle",
        "yellow_block",
        "dark_bottle",
        "red_block",
        "orange_bottle",
    )
    node._item_catalog = lambda: SimpleNamespace(
        items={item_id: object() for item_id in order}
    )

    def view(name, visible):
        return {
            "view_name": name,
            "offset_from_start_m": 0.0,
            "capture": {"color_width": 640},
            "label_match": {
                "diagnostics": {
                    "detections": [
                        {
                            "item_id": item_id,
                            "confidence": 0.9,
                            "bbox_xywh": [40 + (100 * index), 50, 60, 80],
                            "method": "test",
                        }
                        for index, item_id in enumerate(visible)
                    ]
                }
            },
        }

    views = [
        view("left", order[:4]),
        view("right", order[2:]),
    ]
    views[1]["label_match"]["diagnostics"]["detections"].append(
        {
            "item_id": "blue_block",
            "confidence": 0.99,
            "bbox_xywh": [0, 170, 52, 54],
            "method": "color_shape_partial",
        }
    )

    fused = PipelineOrchestratorNode._fuse_single_pass_label_map(
        node,
        views=views,
        item_id="yellow_block",
    )

    assert fused["detected_item_ids_left_to_right"] == list(order)
    assert len(fused["rejected_observations"]) == 1
    assert fused["rejected_observations"][0]["item_id"] == "blue_block"
