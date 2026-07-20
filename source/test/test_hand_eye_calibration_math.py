from pathlib import Path
import importlib.util
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "calibrate_hand_eye",
    PROJECT_ROOT / "scripts" / "calibrate_hand_eye.py",
)
calibrate_hand_eye = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(calibrate_hand_eye)


def test_solve_passes_base_to_tcp_as_gripper_to_base(monkeypatch):
    captured = {}

    def fake_calibrate_hand_eye(rotations, translations, target_rotations, target_translations, *, method):
        captured.setdefault("rotations", rotations)
        captured.setdefault("translations", translations)
        return np.eye(3), np.zeros((3, 1))

    monkeypatch.setattr(calibrate_hand_eye.cv2, "calibrateHandEye", fake_calibrate_hand_eye)

    rotation = calibrate_hand_eye.tcp_to_matrix((100.0, 0.0, 0.0), (0.0, 0.0, 90.0))[:3, :3]
    translation = np.array([0.1, 0.2, 0.3])
    calibrate_hand_eye.solve(
        R_c2t=[np.eye(3), np.eye(3), np.eye(3)],
        t_c2t=[np.zeros(3), np.zeros(3), np.zeros(3)],
        R_b2tcp=[rotation, np.eye(3), np.eye(3)],
        t_b2tcp=[translation, np.zeros(3), np.zeros(3)],
    )

    assert np.allclose(captured["rotations"][0], rotation)
    assert np.allclose(captured["translations"][0], translation)
