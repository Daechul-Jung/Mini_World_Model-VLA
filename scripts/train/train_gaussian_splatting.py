"""
Train a small 3D Gaussian Splatting room model from TUM RGB-D frames.

This is intentionally compact for study:
  1. Initialize a point cloud by back-projecting depth pixels using TUM intrinsics.
  2. Create GaussianScene from those points and RGB colors.
  3. Render random training views with gsplat and optimize RGB reconstruction loss.

Example:
    python3 scripts/train/train_gaussian_splatting.py \
        --sequence_dir data/tum_rgbd/fr1_desk \
        --steps 1000 --max_points 50000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.world_model.splatting.gaussians import (  # noqa: E402
    GaussianScene,
    c2w_to_viewmat,
    make_intrinsics,
    tum_pose_to_c2w,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_dir", required=True, help="TUM sequence dir, e.g. data/tum_rgbd/fr1_desk")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max_points", type=int, default=50_000)
    p.add_argument("--init_frame_stride", type=int, default=30)
    p.add_argument("--train_frame_stride", type=int, default=5)
    p.add_argument("--point_stride", type=int, default=8, help="Use every Nth depth pixel during init")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=Path, default=Path("checkpoints/gaussian_splatting"))
    return p.parse_args()


def load_poses(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices, timestamps, poses = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = [float(v) for v in line.split()]
        if len(values) == 9:
            frame_indices.append(int(values[0]))
            timestamps.append(values[1])
            poses.append(values[2:9])
        else:
            frame_indices.append(len(frame_indices))
            timestamps.append(values[0])
            poses.append(values[1:8])
    return (
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(poses, dtype=np.float32),
    )


def load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_depth(path: Path, size: int, depth_scale: float) -> np.ndarray:
    depth = Image.open(path).resize((size, size), Image.NEAREST)
    return np.asarray(depth, dtype=np.float32) / depth_scale


def scaled_intrinsics(intr: dict, size: int, original_w: int = 640, original_h: int = 480) -> dict:
    sx = size / original_w
    sy = size / original_h
    return {
        "fx": intr["fx"] * sx,
        "fy": intr["fy"] * sy,
        "cx": intr["cx"] * sx,
        "cy": intr["cy"] * sy,
        "depth_scale": intr["depth_scale"],
    }


def backproject_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intr: dict,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = (z > 0.1) & (z < 6.0)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xs - intr["cx"]) / intr["fx"] * z
    y = (ys - intr["cy"]) / intr["fy"] * z
    cam_points = np.stack([x, y, z, np.ones_like(z)], axis=-1)

    pose_t = torch.from_numpy(pose[None])
    c2w = tum_pose_to_c2w(pose_t).squeeze(0).numpy()
    world_points = (c2w @ cam_points.T).T[:, :3]
    colors = rgb[ys.astype(np.int64), xs.astype(np.int64)]
    return world_points.astype(np.float32), (colors * 255.0).astype(np.uint8)


def initialize_scene(args, rgb_paths, depth_paths, poses, intr) -> GaussianScene:
    xyz_parts, rgb_parts = [], []
    for idx in range(0, len(rgb_paths), args.init_frame_stride):
        rgb = load_rgb(rgb_paths[idx], args.image_size)
        depth = load_depth(depth_paths[idx], args.image_size, intr["depth_scale"])
        xyz, colors = backproject_depth(rgb, depth, poses[idx], intr, args.point_stride)
        xyz_parts.append(xyz)
        rgb_parts.append(colors)

    xyz = np.concatenate(xyz_parts, axis=0)
    colors = np.concatenate(rgb_parts, axis=0)
    if len(xyz) > args.max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xyz), size=args.max_points, replace=False)
        xyz = xyz[keep]
        colors = colors[keep]

    log.info("Initialized %d Gaussians from RGB-D backprojection", len(xyz))
    return GaussianScene.from_colmap_points(xyz, colors, sh_degree=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.steps > 0 and device.type != "cuda":
        raise SystemExit("gsplat training requires a CUDA device. Use --steps 0 only for initialization checks.")
    seq_dir = Path(args.sequence_dir)
    rgb_paths = sorted((seq_dir / "rgb_frames").glob("*.png"))
    depth_paths = sorted((seq_dir / "depth_frames").glob("*.png"))
    if not rgb_paths or not depth_paths:
        raise SystemExit(
            "Missing rgb_frames/depth_frames. Re-run scripts/download/download_tum_rgbd.py "
            "or export frames for this sequence."
        )

    frame_indices, _, poses_np = load_poses(seq_dir / "poses.txt")
    frame_indices = frame_indices[
        (frame_indices >= 0) & (frame_indices < min(len(rgb_paths), len(depth_paths)))
    ]
    poses_np = poses_np[: len(frame_indices)]
    rgb_paths = [rgb_paths[i] for i in frame_indices]
    depth_paths = [depth_paths[i] for i in frame_indices]
    n = len(poses_np)

    intr = scaled_intrinsics(json.loads((seq_dir / "intrinsics.json").read_text()), args.image_size)
    scene = initialize_scene(args, rgb_paths, depth_paths, poses_np, intr).to(device)

    train_indices = list(range(0, n, args.train_frame_stride))
    poses = torch.from_numpy(poses_np).to(device)
    optimizer = torch.optim.Adam(scene.parameters(), lr=args.lr)
    K = make_intrinsics(
        intr["fx"], intr["fy"], intr["cx"], intr["cy"], args.batch_size, device=device
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Training 3DGS on %d views for %d steps", len(train_indices), args.steps)

    for step in range(args.steps):
        batch_ids = np.random.choice(train_indices, size=args.batch_size, replace=True)
        target_np = np.stack([load_rgb(rgb_paths[i], args.image_size) for i in batch_ids], axis=0)
        target = torch.from_numpy(target_np).to(device)

        c2w = tum_pose_to_c2w(poses[batch_ids])
        viewmats = c2w_to_viewmat(c2w)
        pred, _, _ = scene.render(viewmats, K, args.image_size, args.image_size)
        pred_rgb = pred[..., :3].clamp(0, 1)

        loss = F.mse_loss(pred_rgb, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            psnr = -10.0 * torch.log10(loss.detach().clamp_min(1e-8))
            log.info("step=%04d loss=%.6f psnr=%.2f", step, loss.item(), psnr.item())

    out = args.out_dir / f"{seq_dir.name}_gaussians.pt"
    torch.save({"scene": scene.state_dict(), "intrinsics": intr, "image_size": args.image_size}, out)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
