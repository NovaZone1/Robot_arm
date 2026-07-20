"""Pure-PyTorch CPU-only reimplementation of pointnet2 CUDA kernels.

Provides the exact same function signatures as the original `pointnet2._ext`
C++/CUDA extension, so `pointnet2_utils.py` can import and call them unchanged.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# furthest_point_sample
# ---------------------------------------------------------------------------
def furthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative furthest point sampling.  (B, N, 3) -> (B, npoint) int64."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.int64, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.int64, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distances = torch.min(distances, dist)
        farthest = distances.argmax(dim=-1)
    return idx


# ---------------------------------------------------------------------------
# gather_operation
# ---------------------------------------------------------------------------
def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather features by index.  (B, C, N) + (B, npoint) -> (B, C, npoint)
    Also handles 3D idx: (B, C, N) + (B, npoint, nsample) -> (B, C, npoint, nsample)"""
    B, C, N = features.shape
    if idx.dim() == 2:
        B2, npoint = idx.shape
        return features.gather(dim=2, index=idx.unsqueeze(1).expand(B, C, npoint).to(torch.int64))
    elif idx.dim() == 3:
        B2, npoint, nsample = idx.shape
        return features.gather(dim=2, index=idx.reshape(B, 1, -1).expand(B, C, -1).to(torch.int64)).reshape(B, C, npoint, nsample)
    raise ValueError(f"idx must be 2D or 3D, got {idx.dim()}D")


def gather_points_grad(grad_out: torch.Tensor, idx: torch.Tensor, N: int) -> torch.Tensor:
    """Reverse of gather_points."""
    B, C = grad_out.shape[:2]
    device = grad_out.device
    grad_features = torch.zeros(B, C, N, device=device, dtype=grad_out.dtype)
    if idx.dim() == 2:
        idx2 = idx.unsqueeze(1).expand(B, C, -1)
        grad_features.scatter_add_(2, idx2.to(torch.int64), grad_out)
    else:
        flat_idx = idx.reshape(B, 1, -1).expand(B, C, -1)
        grad_features.scatter_add_(2, flat_idx.to(torch.int64), grad_out.reshape(B, C, -1))
    return grad_features


# ---------------------------------------------------------------------------
# three_nn + three_interpolate
# ---------------------------------------------------------------------------
def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """Find 3 nearest neighbors of unknown in known.
    (B, n, 3) + (B, m, 3) -> (B, n, 3) dist, (B, n, 3) idx"""
    B, n, _ = unknown.shape
    m = known.shape[1]
    dist2 = ((unknown.unsqueeze(2) - known.unsqueeze(1)) ** 2).sum(dim=-1)
    topk_d2, topk_idx = torch.topk(dist2, k=min(3, m), dim=2, largest=False)
    if m < 3:
        pad_w = 3 - m
        topk_d2 = F.pad(topk_d2, (0, pad_w), value=1e10)
        topk_idx = F.pad(topk_idx, (0, pad_w), value=0)
    return torch.sqrt(topk_d2 + 1e-10), topk_idx.to(torch.int64)


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted interpolation.  (B, C, m) + (B, n, 3) + (B, n, 3) -> (B, C, n)"""
    B, C, m = features.shape
    n = idx.shape[1]
    idx_flat = idx.unsqueeze(1).expand(B, C, -1, -1).reshape(B, C, -1)
    gathered = features.gather(dim=2, index=idx_flat.to(torch.int64)).reshape(B, C, n, 3)
    return (gathered * weight.unsqueeze(1)).sum(dim=-1)


def three_interpolate_grad(grad_out: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, m: int) -> torch.Tensor:
    """Backward of three_interpolate."""
    B, C, n = grad_out.shape
    device = grad_out.device
    weighted_grad = grad_out.unsqueeze(-1) * weight.unsqueeze(1)
    idx_flat = idx.unsqueeze(1).expand(B, C, -1, -1).reshape(B, C, -1)
    grad_features = torch.zeros(B, C, m, device=device, dtype=grad_out.dtype)
    grad_features.scatter_add_(2, idx_flat.to(torch.int64), weighted_grad.reshape(B, C, -1))
    return grad_features


# ---------------------------------------------------------------------------
# group_points
# ---------------------------------------------------------------------------
def group_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Alias for gather_points with 3D idx."""
    return gather_points(features, idx)


def group_points_grad(grad_out: torch.Tensor, idx: torch.Tensor, N: int) -> torch.Tensor:
    """Backward of group_points."""
    return gather_points_grad(grad_out, idx, N)


# ---------------------------------------------------------------------------
# ball_query
# ---------------------------------------------------------------------------
def _ball_query_batch(new_xyz, xyz, radius, nsample):
    npoint, N = new_xyz.shape[0], xyz.shape[0]
    dist2 = ((new_xyz.unsqueeze(1) - xyz.unsqueeze(0)) ** 2).sum(dim=-1)
    dist2[dist2 > radius * radius] = 1e10
    k = min(nsample, N)
    _, group_idx = torch.topk(dist2, k=k, dim=-1, largest=False)
    if k < nsample:
        group_idx = F.pad(group_idx, (0, nsample - k), value=0)
    return group_idx.to(torch.int64)


def ball_query(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """Ball query.  (B, npoint, 3) + (B, N, 3) -> (B, npoint, nsample) int64"""
    idx = [_ball_query_batch(new_xyz[b], xyz[b], radius, nsample) for b in range(xyz.shape[0])]
    return torch.stack(idx, dim=0)


# ---------------------------------------------------------------------------
# cylinder_query
# ---------------------------------------------------------------------------
def cylinder_query(new_xyz: torch.Tensor, xyz: torch.Tensor, rot: torch.Tensor,
                   radius: float, hmin: float, hmax: float, nsample: int) -> torch.Tensor:
    """Cylinder query in rotated local frame.
    (B,npoint,3) + (B,N,3) + (B,npoint,9) -> (B, npoint, nsample) int64

    rot is (B, npoint, 9): flattened 3x3 rotation matrices.
    X axis = cylinder axis (grasp approach direction)."""
    B, npoint, _ = new_xyz.shape
    N = xyz.shape[1]
    device = xyz.device
    rot_mat = rot.reshape(B, npoint, 3, 3)
    rel_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)  # (B, npoint, N, 3)
    rel_local = torch.matmul(rot_mat, rel_xyz.transpose(2, 3)).transpose(2, 3)
    rx, ry, rz = rel_local[..., 0], rel_local[..., 1], rel_local[..., 2]
    valid = (ry ** 2 + rz ** 2 <= radius * radius) & (rx >= hmin) & (rx <= hmax)
    dist2 = torch.where(valid, ry ** 2 + rz ** 2, torch.tensor(1e10, device=device))
    k = min(nsample, N)
    _, topk_idx = torch.topk(dist2, k=k, dim=-1, largest=False)
    if k < nsample:
        topk_idx = F.pad(topk_idx, (0, nsample - k), value=0)
    return topk_idx.to(torch.int64)
