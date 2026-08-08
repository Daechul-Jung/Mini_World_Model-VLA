"""Static-image dataset for stage A.

Stage A is the only world-model stage that can be trained on still images, which
is why LSUN rooms are useful here and useless everywhere else: a latent action
model and a dynamics model both need *ordered frames*, and LSUN is an unordered
pile of photographs of different rooms.

Practical consequence for this project: train the tokenizer on LSUN + TUM frames
together (more visual variety, better codebook coverage), then train stages B/C
on TUM video only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from world_model.core.registry import WM_DATASETS

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(root: Path | str, exclude: Sequence[str] = ("depth",)) -> List[Path]:
    """All images under `root`, skipping paths containing any `exclude` token.

    The default exclusion matters for TUM RGB-D, whose sequence directories hold
    `depth_frames/` alongside `rgb_frames/`; training a tokenizer on 16-bit depth
    maps rendered as images is a silent, hard-to-spot data bug.
    """
    root = Path(root)
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.suffix.lower() in IMAGE_SUFFIXES
        and not any(token in str(p.relative_to(root)).lower() for token in exclude)
    ]
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files


@WM_DATASETS.register("images", note="stage A only; no temporal structure")
class ImageFolderDataset(Dataset):
    """Flat recursive image folder -> (3, H, W) tensors in [-1, 1]."""

    def __init__(
        self,
        roots: str | Sequence[str],
        image_size: int = 128,
        exclude: Sequence[str] = ("depth",),
        augment: bool = True,
        limit: int | None = None,
    ):
        from torchvision import transforms

        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.paths: List[Path] = []
        for root in roots:
            self.paths.extend(list_images(root, exclude))
        if limit:
            self.paths = self.paths[:limit]

        ops = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
        if augment:
            ops = [
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
            ]
        ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"frames": self.transform(img)}
