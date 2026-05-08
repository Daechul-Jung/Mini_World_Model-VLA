"""
Prepare a small ImageNet subset for image diffusion experiments.

ImageNet cannot be directly downloaded by a script in the same way as LSUN/TUM;
you must accept ImageNet's terms and place the official train archive or an
extracted ImageFolder tree on disk first. This script then copies a small,
balanced subset into this project's data directory.

Examples:
    python3 scripts/download/prepare_imagenet_subset.py \
        --imagenet_root /path/to/imagenet/train \
        --output_dir data/imagenet_subset \
        --classes 50 --images_per_class 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet_root", required=True, help="Extracted ImageNet train folder")
    p.add_argument("--output_dir", default="data/imagenet_subset")
    p.add_argument("--classes", type=int, default=50)
    p.add_argument("--images_per_class", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    src_root = Path(args.imagenet_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in sorted(src_root.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class folders found under {src_root}")

    chosen_classes = rng.sample(class_dirs, k=min(args.classes, len(class_dirs)))
    total = 0
    for class_dir in chosen_classes:
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png"):
            images.extend(class_dir.glob(ext))
        if not images:
            continue
        chosen_images = rng.sample(images, k=min(args.images_per_class, len(images)))
        dst = out_root / class_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for path in chosen_images:
            shutil.copy2(path, dst / path.name)
        total += len(chosen_images)
        print(f"{class_dir.name}: {len(chosen_images)} images")

    print(f"Prepared {total} images in {out_root.resolve()}")


if __name__ == "__main__":
    main()
