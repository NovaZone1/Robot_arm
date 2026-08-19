from __future__ import annotations

from datetime import datetime
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

GRASP_ARRAY_LEN = 17


def _grasp_nms(grasp_array, translation_thresh=0.03, rotation_thresh=30.0 / 180.0 * np.pi):
    """Pure-NumPy grasp NMS — sorts by score, suppresses overlapping poses."""
    if len(grasp_array) <= 1:
        return grasp_array.copy()
    scores = grasp_array[:, 0]
    order = np.argsort(-scores)
    sorted_arr = grasp_array[order]
    translations = sorted_arr[:, 13:16]
    rotations = sorted_arr[:, 4:13].reshape(-1, 3, 3)
    n = len(sorted_arr)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        t_dist = np.linalg.norm(translations[i + 1 :] - translations[i], axis=1)
        R_i_T = rotations[i].T
        R_rel = np.matmul(rotations[i + 1 :], R_i_T)
        trace = np.trace(R_rel, axis1=1, axis2=2)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        rot_dist = np.arccos(cos_angle)
        suppress = (t_dist <= translation_thresh) & (rot_dist <= rotation_thresh)
        keep[i + 1 :] &= ~suppress
    return sorted_arr[keep]


class _MinimalGraspGroup:
    """Thin GraspGroup replacement — same interface, no graspnetAPI / autolab_core deps."""

    def __init__(self, array=None):
        if array is None:
            self.grasp_group_array = np.zeros((0, GRASP_ARRAY_LEN), dtype=np.float64)
        elif isinstance(array, np.ndarray):
            self.grasp_group_array = array.astype(np.float64, copy=False)
        elif isinstance(array, str):
            self.grasp_group_array = np.load(array).astype(np.float64)
        else:
            raise TypeError(f"expected ndarray or str, got {type(array)}")

    def __len__(self):
        return len(self.grasp_group_array)

    def __getitem__(self, index):
        if isinstance(index, int):
            if not (-len(self) <= index < len(self)):
                raise IndexError(f"grasp index {index} out of range for group of size {len(self)}")
            return _MinimalGraspGroup(self.grasp_group_array[index : index + 1])
        if isinstance(index, (slice, list, np.ndarray)):
            return _MinimalGraspGroup(self.grasp_group_array[index])
        raise TypeError(f"unsupported index type: {type(index)}")

    def __repr__(self):
        return f"GraspGroup(n={len(self)}, top_scores={self.grasp_group_array[:3, 0].tolist()})"

    @property
    def scores(self):
        return self.grasp_group_array[:, 0]

    @property
    def translations(self):
        return self.grasp_group_array[:, 13:16]

    @property
    def rotation_matrices(self):
        return self.grasp_group_array[:, 4:13].reshape(-1, 3, 3)

    @property
    def widths(self):
        return self.grasp_group_array[:, 1]

    @property
    def heights(self):
        return self.grasp_group_array[:, 2]

    @property
    def depths(self):
        return self.grasp_group_array[:, 3]

    # Singular-named accessors for per-grasp iteration compatibility.
    # When __getitem__ returns a single-row group (int index or iteration),
    # these expose the expected graspnetAPI-like scalar/vector attributes.

    @property
    def score(self):
        return float(self.grasp_group_array[0, 0])

    @property
    def width(self):
        return float(self.grasp_group_array[0, 1])

    @property
    def height(self):
        return float(self.grasp_group_array[0, 2])

    @property
    def depth(self):
        return float(self.grasp_group_array[0, 3])

    @property
    def translation(self):
        return np.asarray(self.grasp_group_array[0, 13:16], dtype=np.float64)

    @property
    def rotation_matrix(self):
        return np.asarray(self.grasp_group_array[0, 4:13], dtype=np.float64).reshape(3, 3)

    def nms(self, translation_thresh=0.03, rotation_thresh=30.0 / 180.0 * np.pi):
        return _MinimalGraspGroup(_grasp_nms(self.grasp_group_array, translation_thresh, rotation_thresh))

    def sort_by_score(self, reverse=False):
        idx = np.argsort(self.grasp_group_array[:, 0])
        if not reverse:
            idx = idx[::-1]
        self.grasp_group_array = self.grasp_group_array[idx]
        return self

    def transform(self, T):
        rotation = T[:3, :3]
        translation = T[:3, 3]
        self.grasp_group_array[:, 13:16] = (rotation @ self.translations.T).T + translation
        self.grasp_group_array[:, 4:13] = np.matmul(rotation, self.rotation_matrices).reshape(-1, 9)
        return self


def make_center_contact_grasp_group(
    translation_m,
    *,
    score: float = 1.0,
    width_m: float = 0.04,
    height_m: float = 0.02,
    depth_m: float = 0.02,
):
    """Build a single contact-centered grasp without running GraspNet."""
    center = np.asarray(translation_m, dtype=np.float64).reshape(3)
    array = np.zeros((1, GRASP_ARRAY_LEN), dtype=np.float64)
    array[0, 0] = float(score)
    array[0, 1] = float(width_m)
    array[0, 2] = float(height_m)
    array[0, 3] = float(depth_m)
    array[0, 4:13] = np.eye(3, dtype=np.float64).reshape(9)
    array[0, 13:16] = center
    return _MinimalGraspGroup(array)

if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.signedinteger):
            return np.int64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.bool_):
            return np.bool_
        return dtype.type

    np.maximum_sctype = _maximum_sctype


def _unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if not path:
            continue
        norm = os.path.abspath(os.path.expanduser(path))
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _build_graspnet_roots():
    env_root = os.environ.get("GRASPNET_BASELINE_ROOT", "")
    search_bases = [
        _THIS_DIR,
        os.path.dirname(_THIS_DIR),
        os.path.dirname(os.path.dirname(_THIS_DIR)),
        os.getcwd(),
    ]
    candidates = [env_root]
    for base in search_bases:
        candidates.append(os.path.join(base, "graspnet-baseline"))
        candidates.append(os.path.join(base, "graspnet_baseline"))
        base_name = os.path.basename(os.path.normpath(base))
        if base_name in ("graspnet-baseline", "graspnet_baseline"):
            candidates.append(base)
    return [path for path in _unique_paths(candidates) if os.path.isdir(path)]


_GRASPNET_CANDIDATES = _build_graspnet_roots()
_GRASPNET_CHECKPOINT_CANDIDATES = _unique_paths(
    [os.environ.get("GRASPNET_CHECKPOINT", ""), os.path.join(_THIS_DIR, "checkpoint-rs.tar")]
    + [os.path.join(root, "checkpoint-rs.tar") for root in _GRASPNET_CANDIDATES]
    + [os.path.join(root, "checkpoint.tar") for root in _GRASPNET_CANDIDATES]
)

for root in _GRASPNET_CANDIDATES:
    for path in [
        root,
        os.path.join(root, "models"),
        os.path.join(root, "utils"),
        os.path.join(root, "graspnetAPI"),
        os.path.join(root, "pointnet2"),
    ]:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

GRASPNET_AVAILABLE = True
GRASPNET_IMPORT_ERROR = ""
try:
    import_errors = []

    try:
        from models.graspnet import GraspNet, pred_decode
    except Exception as exc_models:
        import_errors.append(f"models.graspnet failed: {exc_models!r}")
        try:
            from graspnet import GraspNet, pred_decode
        except Exception as exc_flat:
            import_errors.append(f"graspnet failed: {exc_flat!r}")
            raise ImportError(" | ".join(import_errors)) from exc_flat

    try:
        from utils.collision_detector import ModelFreeCollisionDetector
    except Exception as exc_utils:
        import_errors.append(f"utils.collision_detector failed: {exc_utils!r}")
        try:
            from collision_detector import ModelFreeCollisionDetector
        except Exception as exc_flat_collision:
            import_errors.append(f"collision_detector failed: {exc_flat_collision!r}")
            raise ImportError(" | ".join(import_errors)) from exc_flat_collision

    GraspGroup = _MinimalGraspGroup
except Exception as exc:
    GRASPNET_AVAILABLE = False
    GRASPNET_IMPORT_ERROR = str(exc)


def resolve_graspnet_checkpoint(checkpoint_path: str = "") -> str:
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
    candidates.extend(_GRASPNET_CHECKPOINT_CANDIDATES)
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized
    return checkpoint_path


class GraspNetRunner:
    def __init__(self, checkpoint_path, device: str = "cuda", num_point: int = 20000, topk: int = 50, voxel_size: float = 0.01, collision_thresh: float = 0.01, approach_dist: float = 0.05):
        if not GRASPNET_AVAILABLE:
            raise ImportError(f"GraspNet modules are not available. Import error: {GRASPNET_IMPORT_ERROR}")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for GraspNet")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"GraspNet checkpoint not found: {checkpoint_path}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)
        self.num_point = int(num_point)
        self.topk = int(topk)
        self.voxel_size = float(voxel_size)
        self.collision_thresh = float(collision_thresh)
        self.approach_dist = float(approach_dist)

        self.net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.net.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def _sample_points(self, points):
        if points is None or len(points) == 0:
            return None
        points = np.asarray(points, dtype=np.float32)
        num_points = len(points)
        if num_points >= self.num_point:
            indices = np.random.choice(num_points, self.num_point, replace=False)
        else:
            indices = np.concatenate([np.arange(num_points), np.random.choice(num_points, self.num_point - num_points, replace=True)], axis=0)
        return points[indices].astype(np.float32)

    def predict(self, scene_points, object_points):
        if object_points is None or len(object_points) < 32:
            return None
        sampled = self._sample_points(object_points)
        if sampled is None:
            return None
        end_points = {"point_clouds": torch.from_numpy(sampled[np.newaxis].astype(np.float32)).to(self.device)}
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        if len(grasp_group) == 0:
            return grasp_group
        if scene_points is not None and len(scene_points) > 0:
            try:
                detector = ModelFreeCollisionDetector(np.asarray(scene_points, dtype=np.float32), voxel_size=self.voxel_size)
                collision_mask = detector.detect(grasp_group, approach_dist=self.approach_dist, collision_thresh=self.collision_thresh)
                grasp_group = grasp_group[~collision_mask]
            except Exception:
                pass
        if len(grasp_group) == 0:
            return grasp_group
        try:
            grasp_group = grasp_group.nms()
        except Exception:
            pass
        try:
            grasp_group = grasp_group.sort_by_score()
        except Exception:
            pass
        try:
            grasp_group = grasp_group[: self.topk]
        except Exception:
            pass
        return grasp_group

    @staticmethod
    def subset_grasp_group(grasp_group, indices):
        if grasp_group is None:
            return None
        indices_np = np.asarray(indices, dtype=np.int64)
        if indices_np.size == 0:
            try:
                return grasp_group[:0]
            except Exception:
                return None
        try:
            return grasp_group[indices_np]
        except Exception:
            array = grasp_group_to_array(grasp_group)
            if array.size == 0:
                return None
            subset = array[indices_np]
            try:
                return GraspGroup(subset)
            except Exception:
                return subset

    @staticmethod
    def merge_grasp_groups(*grasp_groups, topk: int | None = None):
        arrays = []
        for grasp_group in grasp_groups:
            if grasp_group is None or len(grasp_group) == 0:
                continue
            array = grasp_group_to_array(grasp_group)
            if array.ndim == 2 and array.shape[0] > 0:
                arrays.append(array)
        if not arrays:
            return None
        merged = GraspGroup(np.concatenate(arrays, axis=0))
        try:
            merged = merged.nms()
        except Exception:
            pass
        try:
            merged = merged.sort_by_score()
        except Exception:
            pass
        if topk is not None:
            try:
                merged = merged[: int(topk)]
            except Exception:
                pass
        return merged


def grasp_group_to_array(grasp_group):
    if grasp_group is None:
        return np.zeros((0, 0), dtype=np.float32)
    if hasattr(grasp_group, "grasp_group_array"):
        return np.asarray(grasp_group.grasp_group_array)
    try:
        return np.asarray(grasp_group)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)


def save_grasp_results(grasp_groups, text_prompt: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = text_prompt.replace(" ", "_")[:20]
    saved_files = []
    summary_lines = []
    for index, grasp_group in enumerate(grasp_groups):
        if grasp_group is None or len(grasp_group) == 0:
            summary_lines.append(f"instance {index}: no valid grasp")
            continue
        array = grasp_group_to_array(grasp_group)
        npy_path = os.path.join(output_dir, f"grasps_{tag}_{index}_{timestamp}.npy")
        np.save(npy_path, array)
        saved_files.append(npy_path)
        summary_lines.append(f"instance {index}: {len(grasp_group)} grasp(s)")

    txt_path = os.path.join(output_dir, f"grasps_{tag}_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))
    saved_files.append(txt_path)
    return saved_files
