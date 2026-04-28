"""
Download TUM RGB-D dataset sequences for world model training and 3DGS reconstruction.

TUM RGB-D (Sturm et al., 2012) provides:
  - Synchronized RGB + depth frames
  - Ground-truth camera trajectories (from motion capture)
  - Various indoor environments: office, desk, room, plant scenes

Perfect for:
  1. World model training — video sequences of room-scale environments
  2. 3D Gaussian Splatting — SLAM poses + RGB frames → 3D room reconstruction
  3. SLAM evaluation — compare your SLAM vs ground truth

Dataset page: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

Usage:
    # Download recommended sequences for room generation
    python scripts/download/download_tum_rgbd.py \
        --sequences fr1_desk fr2_desk fr3_office \
        --output_dir data/tum_rgbd

Recommended sequences for room/office reconstruction:
  fr1_desk      — office desk, ~30s, 9.1 MB
  fr2_desk      — larger desk scene, ~120s, 56 MB
  fr3_office    — office room overview, ~60s, 88 MB
  fr1_room      — full room pan, ~30s, 14 MB
  fr2_large_no_loop — large indoor space, ~340s, 340 MB
"""

import argparse
import hashlib
import os
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# Official TUM RGB-D sequence URLs
# Format: sequence_name → (url, description)
SEQUENCES: Dict[str, tuple] = {
    # freiburg1 (close range, ~0.5m focal length)
    "fr1_desk":       ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
                       "Office desk, 9.1 MB, ~580 frames"),
    "fr1_desk2":      ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
                       "Office desk variant, 11 MB"),
    "fr1_room":       ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz",
                       "Full room pan, 14 MB, ~1350 frames"),
    "fr1_plant":      ("https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
                       "Potted plant, 8 MB"),
    # freiburg2 (medium range, ~0.9m focal length)
    "fr2_desk":       ("https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
                       "Larger desk scene, 56 MB, ~2900 frames"),
    "fr2_dishes":     ("https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_dishes.tgz",
                       "Kitchen dishes, 52 MB"),
    "fr2_large_no_loop": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_no_loop.tgz",
                           "Large indoor space, 340 MB"),
    # freiburg3 (medium range, structured light depth)
    "fr3_office":     ("https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
                       "Office room overview, 88 MB, ~2500 frames"),
    "fr3_nostructure_texture": ("https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz",
                                 "Textured objects near, 60 MB"),
}

# Camera intrinsics for each freiburg sequence set
INTRINSICS = {
    "fr1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "depth_scale": 5000.0},
    "fr2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7, "depth_scale": 5208.0},
    "fr3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "depth_scale": 5000.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+",
                   default=["fr1_desk", "fr2_desk", "fr3_office"],
                   choices=list(SEQUENCES.keys()))
    p.add_argument("--output_dir", default="data/tum_rgbd")
    p.add_argument("--extract_frames", action="store_true", default=True,
                   help="Extract RGB frames and depth maps to separate folders")
    p.add_argument("--image_size", type=int, default=None,
                   help="Resize RGB to square (None = keep original 640×480)")
    return p.parse_args()


def _reporthook(count: int, block_size: int, total_size: int) -> None:
    mb_done = count * block_size / 1e6
    mb_total = total_size / 1e6
    print(f"\r  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)


def download_sequence(seq_name: str, out_dir: Path) -> Path:
    url, desc = SEQUENCES[seq_name]
    tgz_path = out_dir / f"{seq_name}.tgz"
    seq_dir = out_dir / seq_name

    if seq_dir.exists():
        print(f"  {seq_name} already extracted at {seq_dir}")
        return seq_dir

    if not tgz_path.exists():
        print(f"\nDownloading {seq_name}: {desc}")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, str(tgz_path), reporthook=_reporthook)
        print()

    print(f"  Extracting {tgz_path.name}...")
    import tarfile
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(out_dir)

    tgz_path.unlink()
    # TUM archives extract to a folder named rgbd_dataset_freiburgX_name
    extracted = list(out_dir.glob("rgbd_dataset_*"))
    if extracted:
        extracted[0].rename(seq_dir)
    return seq_dir


def parse_associations(seq_dir: Path) -> List[Dict]:
    """
    Parse rgb.txt, depth.txt, and groundtruth.txt into aligned frame list.

    Returns list of dicts:
      {"timestamp": float, "rgb": Path, "depth": Path, "pose": (tx,ty,tz,qx,qy,qz,qw)}
    """
    def read_file_list(path: Path) -> Dict[float, str]:
        entries = {}
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            entries[float(parts[0])] = parts[1]
        return entries

    rgb_list = read_file_list(seq_dir / "rgb.txt")
    depth_list = read_file_list(seq_dir / "depth.txt")

    # Load ground truth poses
    pose_list: Dict[float, tuple] = {}
    gt_path = seq_dir / "groundtruth.txt"
    if gt_path.exists():
        for line in gt_path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            ts = float(parts[0])
            pose_list[ts] = tuple(float(x) for x in parts[1:8])

    # Temporally align: for each RGB frame, find nearest depth and pose
    frames = []
    for rgb_ts, rgb_path in sorted(rgb_list.items()):
        # Nearest depth
        if not depth_list:
            continue
        depth_ts = min(depth_list.keys(), key=lambda t: abs(t - rgb_ts))
        if abs(depth_ts - rgb_ts) > 0.02:   # 20ms tolerance
            continue

        # Nearest pose (optional)
        pose = None
        if pose_list:
            pose_ts = min(pose_list.keys(), key=lambda t: abs(t - rgb_ts))
            if abs(pose_ts - rgb_ts) < 0.05:
                pose = pose_list[pose_ts]

        frames.append({
            "timestamp": rgb_ts,
            "rgb": seq_dir / rgb_path,
            "depth": seq_dir / depth_list[depth_ts],
            "pose": pose,
        })

    return frames


def export_frames(seq_dir: Path, image_size: Optional[int]) -> None:
    """
    Export RGB frames and depth maps as PNG images into rgb_frames/ and depth_frames/.
    Also write a poses.txt file with camera poses (for Gaussian Splatting).
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise SystemExit("Run: pip install pillow numpy")

    frames = parse_associations(seq_dir)
    if not frames:
        print(f"  Could not parse frame associations in {seq_dir}")
        return

    rgb_out = seq_dir / "rgb_frames"
    rgb_out.mkdir(exist_ok=True)
    poses_lines = ["# timestamp tx ty tz qx qy qz qw"]

    for i, frame in enumerate(frames):
        rgb = Image.open(frame["rgb"]).convert("RGB")
        if image_size:
            rgb = rgb.resize((image_size, image_size), Image.LANCZOS)
        rgb.save(rgb_out / f"{i:05d}.png")

        if frame["pose"] is not None:
            poses_lines.append(f"{frame['timestamp']:.6f} " + " ".join(f"{v:.8f}" for v in frame["pose"]))

        if i % 200 == 0:
            print(f"  Exported {i+1}/{len(frames)} frames")

    (seq_dir / "poses.txt").write_text("\n".join(poses_lines))
    print(f"  Saved {len(frames)} RGB frames to {rgb_out}")
    print(f"  Saved poses.txt ({len(poses_lines)-1} poses)")


def print_next_steps(seq_dir: Path) -> None:
    print(
        f"\nNext steps for 3D Gaussian Splatting with {seq_dir.name}:\n"
        f"  1. Poses are already in poses.txt (from motion capture, no COLMAP needed)\n"
        f"  2. Install gsplat:  pip install gsplat\n"
        f"  3. Run training with src/world_model/splatting/gaussians.py\n"
        f"\nNext steps for world model training:\n"
        f"  1. Use {seq_dir}/rgb_frames/ as video frames\n"
        f"  2. Run: python scripts/train/train_world_model.py --data_dir {seq_dir}\n"
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seq_name in args.sequences:
        print(f"\n{'='*50}")
        print(f"Sequence: {seq_name}")
        seq_dir = download_sequence(seq_name, out)

        if args.extract_frames:
            print(f"  Exporting frames...")
            export_frames(seq_dir, args.image_size)

        # Write camera intrinsics
        prefix = seq_name[:3]   # e.g. "fr1", "fr2", "fr3"
        if prefix in INTRINSICS:
            import json
            intr_path = seq_dir / "intrinsics.json"
            with open(intr_path, "w") as f:
                json.dump(INTRINSICS[prefix], f, indent=2)
            print(f"  Saved camera intrinsics to {intr_path}")

        print_next_steps(seq_dir)


if __name__ == "__main__":
    main()
