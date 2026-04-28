from __future__ import annotations
"""
3D Gaussian Splatting (3DGS) scene representation.

Reference: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023).

This module stores and optimizes a set of 3D Gaussians that represent a static scene.
For the actual differentiable tile-based rasterizer, use the `gsplat` library:

    pip install gsplat

Workflow for a personal room recording:
  1. Record video of your room (phone/camera, slow movement, good lighting).
  2. Run COLMAP to estimate camera poses:       colmap automatic_reconstructor --image_path ./images --workspace_path ./colmap_out
  3. Convert COLMAP output to our format (use read_colmap.py helper).
  4. Initialize Gaussians from COLMAP sparse point cloud.
  5. Train with train_gaussians() alternating densification + pruning.
  6. Render novel views with render_novel_view().

For TUM RGB-D data, camera poses come from the ground-truth trajectory files,
so skip COLMAP and use the provided poses directly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to 3×3 rotation matrices.

    Args:
        q: (N, 4) WXYZ quaternions (must be unit-length)
    Returns:
        R: (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def build_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Build 3D covariance matrices Σ = R S S^T R^T from per-Gaussian
    scale vectors and rotation quaternions.

    Args:
        scales: (N, 3) log-space scales (will be exp'd)
        quats:  (N, 4) rotation quaternions (will be normalized)
    Returns:
        cov3d: (N, 3, 3)
    """
    S = torch.diag_embed(scales.exp())    # (N, 3, 3) diagonal scale matrix
    R = quaternion_to_rotation(nn.functional.normalize(quats, dim=-1))
    RS = R @ S
    return RS @ RS.transpose(-1, -2)      # (N, 3, 3)


class GaussianScene(nn.Module):
    """
    Learnable set of 3D Gaussians representing a scene.

    Each Gaussian is parameterized by:
      - means:      3D position (μ)
      - scales:     log-space axis scales (s_x, s_y, s_z)
      - quats:      rotation as unit quaternion (w, x, y, z)
      - opacities:  logit-space opacity (sigmoid to get α in [0,1])
      - sh_dc:      degree-0 spherical harmonic (base color, per channel)
      - sh_rest:    higher-degree SH coefficients (view-dependent color)

    Args:
        means:    (N, 3) initial 3D positions (from COLMAP or random)
        sh_degree: maximum SH degree for view-dependent color (0=constant color)
    """

    SH_C0 = 0.28209479177387814

    def __init__(self, means: torch.Tensor, sh_degree: int = 3):
        super().__init__()
        N = means.shape[0]
        self.sh_degree = sh_degree
        n_sh_rest = (sh_degree + 1) ** 2 - 1   # total SH coefficients minus DC

        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(torch.zeros(N, 3))           # log-space
        self.quats = nn.Parameter(torch.zeros(N, 4))            # WXYZ; init to identity
        self.quats.data[:, 0] = 1.0
        self.opacities = nn.Parameter(inverse_sigmoid(0.1 * torch.ones(N)))
        self.sh_dc = nn.Parameter(torch.zeros(N, 3, 1))         # (N, 3 channels, 1 coeff)
        self.sh_rest = nn.Parameter(torch.zeros(N, 3, n_sh_rest))

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)

    @property
    def get_scales(self) -> torch.Tensor:
        return torch.exp(self.scales)

    @property
    def get_rotation(self) -> torch.Tensor:
        return nn.functional.normalize(self.quats, dim=-1)

    @property
    def get_covariance(self) -> torch.Tensor:
        return build_covariance(self.scales, self.quats)

    def get_colors(self, view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate per-Gaussian colors using spherical harmonics.

        Args:
            view_dirs: (N, 3) unit vectors from Gaussian to camera (optional)
        Returns:
            colors: (N, 3) RGB in [0, 1]
        """
        color = self.SH_C0 * self.sh_dc[:, :, 0]   # (N, 3) DC component

        if view_dirs is not None and self.sh_degree > 0:
            # Add higher-order SH terms (simplified; use tinycudann or gsplat for full eval)
            color = color  # TODO: add SH evaluation for sh_rest given view_dirs

        return (color + 0.5).clamp(0, 1)

    @classmethod
    def from_colmap_points(
        cls,
        xyz: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        sh_degree: int = 3,
    ) -> "GaussianScene":
        """
        Initialize from a COLMAP sparse point cloud.

        Args:
            xyz: (N, 3) point positions
            rgb: (N, 3) point colors in [0, 255] or None
        """
        means = torch.from_numpy(xyz.astype(np.float32))
        scene = cls(means, sh_degree=sh_degree)

        if rgb is not None:
            # Initialize DC SH to match the point cloud color
            rgb_norm = torch.from_numpy(rgb.astype(np.float32)) / 255.0
            scene.sh_dc.data[:, :, 0] = (rgb_norm - 0.5) / cls.SH_C0

        # Initialize scales from nearest-neighbor distances
        from torch.nn.functional import pdist
        if means.shape[0] > 1:
            dists = torch.cdist(means[:1000], means[:1000])
            dists.fill_diagonal_(float('inf'))
            nn_dist = dists.min(dim=-1).values.mean().clamp(min=1e-4)
            scene.scales.data.fill_(nn_dist.log().item())

        return scene

    # ------------------------------------------------------------------
    # Densification & pruning helpers (called during training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def densify_and_prune(
        self,
        grad_threshold: float = 2e-4,
        opacity_threshold: float = 0.005,
        max_gaussians: int = 1_000_000,
        scene_extent: float = 1.0,
    ) -> int:
        """
        Adaptive control of Gaussians (clone small, split large, prune transparent).
        Returns the number of Gaussians added.

        This is a simplified placeholder; production 3DGS uses accumulated gradients
        tracked via hooks. See the original 3DGS code or gsplat for full implementation.
        """
        prune_mask = self.get_opacities < opacity_threshold

        # Remove pruned Gaussians (in-place parameter replacement)
        keep = ~prune_mask
        for attr in ('means', 'scales', 'quats', 'opacities', 'sh_dc', 'sh_rest'):
            param = getattr(self, attr)
            object.__setattr__(self, attr, nn.Parameter(param.data[keep]))

        n_removed = prune_mask.sum().item()
        return -n_removed

    def forward(self):
        raise NotImplementedError(
            "Use gsplat.rasterization() for differentiable rendering. "
            "Pass self.means, self.get_covariance, self.get_opacities, self.get_colors() "
            "along with camera intrinsics/extrinsics."
        )
