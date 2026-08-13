import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard_module():
    path = PROJECT_ROOT / "scripts" / "run_grasp_dashboard.py"
    spec = importlib.util.spec_from_file_location("run_grasp_dashboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_can_port_follows_environment(monkeypatch):
    monkeypatch.setenv("PIPER_CAN_PORT", "can1")

    dashboard = _load_dashboard_module()

    assert dashboard.CAN_PORT == "can1"


def test_start_live_stack_does_not_launch_duplicate_saved_process(monkeypatch):
    dashboard = _load_dashboard_module()
    monkeypatch.setattr(
        dashboard,
        "_dashboard_stack_process",
        lambda: (1234, {"log": "/tmp/live_stack.log"}),
    )
    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": False, "components": {}},
    )

    result = dashboard._start_live_stack()

    assert result["ok"] is True
    assert result["stage"] == "starting_stack"
    assert result["pid"] == 1234
    assert "不要重复启动" in result["message"]


def test_param_set_serializes_boolean_for_subprocess(monkeypatch):
    dashboard = _load_dashboard_module()
    calls = []

    def run_ros2(args, *, timeout_s):
        calls.append((args, timeout_s))
        return {"ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(dashboard, "_run_ros2_command", run_ros2)

    assert dashboard._param_set("/grasp_pipeline", "enabled", True)["ok"] is True
    assert dashboard._param_set("/grasp_pipeline", "enabled", False)["ok"] is True
    assert calls == [
        (
            ["param", "set", "/grasp_pipeline", "enabled", "true"],
            10.0,
        ),
        (
            ["param", "set", "/grasp_pipeline", "enabled", "false"],
            10.0,
        ),
    ]


def test_param_set_serializes_empty_string_as_yaml(monkeypatch):
    dashboard = _load_dashboard_module()
    calls = []
    monkeypatch.setattr(
        dashboard,
        "_run_ros2_command",
        lambda args, *, timeout_s: (
            calls.append((args, timeout_s))
            or {"ok": True, "stdout": "", "stderr": ""}
        ),
    )

    assert dashboard._param_set("/grasp_pipeline", "prompt", "")["ok"] is True
    assert calls == [
        (["param", "set", "/grasp_pipeline", "prompt", "''"], 10.0)
    ]


def test_boolean_param_verification_requires_matching_readback(monkeypatch):
    dashboard = _load_dashboard_module()
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_get",
        lambda *_args, **_kwargs: {
            "ok": True,
            "stdout": "Boolean value is: False",
            "stderr": "",
        },
    )

    result = dashboard._set_boolean_param_verified(
        "/grasp_pipeline",
        "base_multiview_enabled",
        True,
    )

    assert result["ok"] is False
    assert result["expected"] == "true"
    assert result["actual"] == "false"


def test_boolean_param_verification_accepts_matching_readback(monkeypatch):
    dashboard = _load_dashboard_module()
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_get",
        lambda *_args, **_kwargs: {
            "ok": True,
            "stdout": "Boolean value is: True",
            "stderr": "",
        },
    )

    result = dashboard._set_boolean_param_verified(
        "/grasp_pipeline",
        "base_multiview_enabled",
        True,
    )

    assert result["ok"] is True


def test_start_grasp_uses_explicit_safe_top_down_strategy(monkeypatch):
    dashboard = _load_dashboard_module()
    parameter_calls = []

    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda node, name, value: (
            parameter_calls.append((node, name, value))
            or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_trigger_service",
        lambda *_args, **_kwargs: {"ok": True, "stdout": "accepted"},
    )

    result = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": False,
            "prompt": "red block",
            "execution_strategy": "safe_top_down",
            "use_object_center_contact": True,
            "execute": True,
            "confirm": True,
        }
    )

    assert result["ok"] is True
    assert (
        "/robot_executor",
        "execution_strategy",
        "safe_top_down",
    ) in parameter_calls


def test_place_after_grasp_requires_catalog_item():
    dashboard = _load_dashboard_module()
    result = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": False,
            "prompt": "bottle",
            "target_item_id": "",
            "place_after_grasp": True,
        }
    )
    assert result["ok"] is False
    assert "指定物品" in result["error"]


def test_base_grasp_scan_requires_target_and_motion_ack():
    dashboard = _load_dashboard_module()

    missing_target = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": False,
            "prompt": "blue block",
            "base_grasp_scan": True,
            "base_motion_ack": True,
        }
    )
    missing_ack = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": False,
            "prompt": "blue block",
            "target_item_id": "blue_block",
            "base_grasp_scan": True,
            "base_motion_ack": False,
        }
    )

    assert missing_target["ok"] is False
    assert "指定物品" in missing_target["error"]
    assert missing_ack["ok"] is False
    assert "急停" in missing_ack["error"]


def test_automatic_target_card_allows_empty_manual_target(monkeypatch):
    dashboard = _load_dashboard_module()
    parameter_calls = []
    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda node, name, value: (
            parameter_calls.append((node, name, value)) or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_trigger_service",
        lambda *_args, **_kwargs: {"ok": True, "stdout": "accepted"},
    )

    result = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": True,
            "prompt": "",
            "target_item_id": "",
            "base_motion_ack": True,
        }
    )

    assert result["ok"] is True
    assert ("/grasp_pipeline", "auto_target_from_card", "true") in parameter_calls
    assert ("/grasp_pipeline", "target_item_id", "") in parameter_calls
    assert ("/grasp_pipeline", "base_grasp_scan_enabled", "true") in parameter_calls
    assert not any(
        node == "/robot_executor" and name == "execution_strategy"
        for node, name, _value in parameter_calls
    )


def test_aligned_place_requires_explicit_release_ack():
    dashboard = _load_dashboard_module()

    result = dashboard._execute_aligned_place_from_payload(
        {"target_item_id": "blue_block", "release_ack": False}
    )

    assert result["ok"] is False
    assert "释放" in result["error"]


def test_aligned_place_uses_one_shot_safety_latch(monkeypatch):
    dashboard = _load_dashboard_module()
    parameter_calls = []
    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda node, name, value: (
            parameter_calls.append((node, name, value))
            or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_trigger_service",
        lambda service, timeout_s: {
            "ok": True,
            "stdout": "response: success=True",
            "service": service,
            "timeout_s": timeout_s,
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_placement_scan_payload",
        lambda: {"success": True},
    )

    result = dashboard._execute_aligned_place_from_payload(
        {"target_item_id": "blue_block", "release_ack": True}
    )

    assert result["ok"] is True
    assert (
        "/grasp_pipeline",
        "base_aligned_place_enabled",
        True,
    ) in parameter_calls
    assert parameter_calls[-1] == (
        "/grasp_pipeline",
        "base_aligned_place_enabled",
        False,
    )


def test_place_after_grasp_enables_dynamic_box_localization(monkeypatch):
    dashboard = _load_dashboard_module()
    parameter_calls = []
    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda node, name, value: (
            parameter_calls.append((node, name, value))
            or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_trigger_service",
        lambda *_args, **_kwargs: {"ok": True, "stdout": "accepted"},
    )

    result = dashboard._start_grasp_from_payload(
        {
            "auto_target_from_card": False,
            "prompt": "green bottle",
            "target_item_id": "green_bottle",
            "place_after_grasp": True,
        }
    )

    assert result["ok"] is True
    assert (
        "/grasp_pipeline",
        "dynamic_box_localization",
        "true",
    ) in parameter_calls


def test_scan_placement_returns_fresh_saved_result(monkeypatch, tmp_path):
    dashboard = _load_dashboard_module()
    monkeypatch.setattr(dashboard, "PLACEMENT_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda *_args, **_kwargs: {"ok": True},
    )

    def trigger(*_args, **_kwargs):
        (tmp_path / "latest.json").write_text(
            json.dumps(
                {
                    "scan_id": "placement-scan-test",
                    "success": False,
                    "label_match": {"detected_label_count": 4},
                    "images": {},
                }
            ),
            encoding="utf-8",
        )
        return {"ok": True, "stdout": "Trigger_Response(success=True, message='ok')"}

    monkeypatch.setattr(dashboard, "_trigger_service", trigger)

    result = dashboard._scan_placement_from_payload(
        {"target_item_id": "red_block"}
    )

    assert result["ok"] is True
    assert result["validation_ok"] is False
    assert "4/6" in result["message"]
    assert result["scan"]["scan_id"] == "placement-scan-test"
    assert result["scan"]["available"] is True


def test_multiview_scan_requires_explicit_base_motion_ack():
    dashboard = _load_dashboard_module()
    result = dashboard._scan_placement_multi_view_from_payload(
        {"target_item_id": "red_block", "base_motion_ack": False}
    )

    assert result["ok"] is False
    assert "急停" in result["error"]


def test_target_box_scan_requires_target_and_motion_ack():
    dashboard = _load_dashboard_module()

    no_target = dashboard._scan_and_align_placement_target_from_payload(
        {"base_motion_ack": True}
    )
    no_ack = dashboard._scan_and_align_placement_target_from_payload(
        {"target_item_id": "blue_block", "base_motion_ack": False}
    )

    assert no_target["ok"] is False
    assert "指定物品" in no_target["error"]
    assert no_ack["ok"] is False
    assert "急停" in no_ack["error"]
