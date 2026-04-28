"""
Download a subset of LSUN bedroom and living_room images for image diffusion training.

LSUN contains millions of scene images. We download a small subset (10K–50K images)
sufficient to train a medium-sized LDM for room generation.

Usage:
    # Download 20,000 bedroom + living_room images
    python scripts/download/download_lsun_rooms.py \
        --n_images 20000 \
        --output_dir data/lsun_rooms \
        --categories bedroom living_room

Requirements:
    pip install lmdb pillow torchvision

Notes:
    - LSUN is distributed as LMDB databases. The downloader fetches them from
      the LSUN authors' servers (~10GB per category for full data).
    - We only export the first --n_images from each LMDB (fast, no full download needed).
    - Alternatively, use torchvision.datasets.LSUN after downloading the .zip files from:
      http://dl.yf.io/lsun/scenes/

Alternative (smaller/simpler): Use ADE20K indoor images (~20K images, 900MB):
    wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
"""

import argparse
import io
import os
import subprocess
from pathlib import Path
from typing import List

LSUN_URL_TEMPLATE = "http://dl.yf.io/lsun/scenes/{category}_train_lmdb.zip"

SUPPORTED_CATEGORIES = [
    "bedroom",
    "living_room",
    "kitchen",
    "dining_room",
    "bathroom",
    "classroom",
    "conference_room",
    "restaurant",
    "tower",
    "bridge",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["bedroom", "living_room"],
                   choices=SUPPORTED_CATEGORIES)
    p.add_argument("--n_images", type=int, default=20_000,
                   help="Max images to export per category")
    p.add_argument("--output_dir", default="data/lsun_rooms")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--download", action="store_true", default=False,
                   help="Download LMDB zip (large!). Skips download if LMDB already present.")
    return p.parse_args()


def download_lmdb(category: str, output_dir: Path) -> Path:
    """Download the LMDB zip for a LSUN category."""
    url = LSUN_URL_TEMPLATE.format(category=category)
    zip_path = output_dir / f"{category}_train_lmdb.zip"
    lmdb_path = output_dir / f"{category}_train_lmdb"

    if lmdb_path.exists():
        print(f"  LMDB already exists at {lmdb_path}, skipping download.")
        return lmdb_path

    print(f"  Downloading {url} ...")
    subprocess.run(["wget", "-q", "-O", str(zip_path), url], check=True)
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(output_dir)], check=True)
    zip_path.unlink()
    print(f"  Extracted to {lmdb_path}")
    return lmdb_path


def export_images(
    lmdb_path: Path,
    category: str,
    out_dir: Path,
    n_images: int,
    image_size: int,
) -> int:
    """Export images from LMDB to JPEG files."""
    try:
        import lmdb
        from PIL import Image
    except ImportError:
        raise SystemExit("Run: pip install lmdb pillow")

    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, max_readers=1)
    saved = 0

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if saved >= n_images:
                break
            try:
                img = Image.open(io.BytesIO(val)).convert("RGB")
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(cat_dir / f"{saved:07d}.jpg", quality=90)
                saved += 1
                if saved % 5000 == 0:
                    print(f"    {category}: {saved}/{n_images}")
            except Exception:
                continue

    env.close()
    return saved


def export_from_torchvision(
    category: str,
    out_dir: Path,
    n_images: int,
    image_size: int,
    lsun_root: str = "data/lsun_raw",
) -> int:
    """
    Alternative: use torchvision.datasets.LSUN if LMDB is already downloaded
    to lsun_root via the LSUN downloader script.
    """
    try:
        from torchvision.datasets import LSUN
        from PIL import Image
    except ImportError:
        raise SystemExit("Run: pip install torchvision pillow")

    ds = LSUN(root=lsun_root, classes=[f"{category}_train"])
    out_cat = out_dir / category
    out_cat.mkdir(parents=True, exist_ok=True)

    for i, (img, _) in enumerate(ds):
        if i >= n_images:
            break
        img_resized = img.resize((image_size, image_size), Image.LANCZOS)
        img_resized.save(out_cat / f"{i:07d}.jpg", quality=90)
        if i % 5000 == 0 and i > 0:
            print(f"  {category}: {i}/{n_images}")

    return min(n_images, len(ds))


def print_ade20k_alternative() -> None:
    print(
        "\nAlternative — ADE20K indoor scenes (smaller, ~900MB):\n"
        "  wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip\n"
        "  unzip ADEChallengeData2016.zip\n"
        "Then filter images under images/training/ for indoor categories."
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for cat in args.categories:
        print(f"\nProcessing category: {cat}")

        lmdb_path = out / f"{cat}_train_lmdb"
        if args.download:
            lmdb_path = download_lmdb(cat, out)
        elif not lmdb_path.exists():
            print(f"  LMDB not found at {lmdb_path}.")
            print(f"  Run with --download to fetch it, or manually download from:")
            print(f"  {LSUN_URL_TEMPLATE.format(category=cat)}")
            continue

        n = export_images(lmdb_path, cat, out, args.n_images, args.image_size)
        total += n
        print(f"  Saved {n} images from {cat}")

    print(f"\nTotal images saved: {total}")
    print(f"Output directory: {out.resolve()}")
    print_ade20k_alternative()


if __name__ == "__main__":
    main()
