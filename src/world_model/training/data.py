"""Dataloader construction shared by all four world-model stages.

One place decides how `cfg["data"]` becomes loaders, so a stage never grows its
own dataset-splitting logic and the train/val split is identical across stages
(which matters: a val clip that was in stage A's training set makes stage C's
validation optimistic).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from world_model.core.registry import WM_DATASETS


def build_dataset(cfg: Dict[str, Any]) -> Dataset:
    """`cfg` is a registry config: `{"name": "video_folder", "roots": [...], ...}`."""
    return WM_DATASETS.build(cfg)


def split_dataset(
    dataset: Dataset, val_fraction: float = 0.05, seed: int = 0, max_val: int = 512
) -> Tuple[Dataset, Optional[Dataset]]:
    """Deterministic split by index.

    Note this is a *random* split, so clips from the same sequence land on both
    sides. For a strict generalisation number, hold out whole sequences instead
    by listing them separately under `data.val`.
    """
    n = len(dataset)
    n_val = min(int(n * val_fraction), max_val)
    if n_val < 1:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])


def build_loaders(
    cfg: Dict[str, Any], batch_size: int, seed: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val loaders from `cfg["data"]`.

    `data.val` (optional) is a full dataset config for a held-out set. When
    absent, a fraction of the training set is split off.
    """
    data_cfg = dict(cfg["data"])
    loader_cfg = {
        "num_workers": data_cfg.pop("num_workers", 4),
        "pin_memory": data_cfg.pop("pin_memory", True),
        "persistent_workers": data_cfg.pop("persistent_workers", False),
        "drop_last": True,
    }
    val_cfg = data_cfg.pop("val", None)
    val_fraction = data_cfg.pop("val_fraction", 0.05)

    train_ds = build_dataset(data_cfg)
    if val_cfg:
        val_ds: Optional[Dataset] = build_dataset(val_cfg)
    else:
        train_ds, val_ds = split_dataset(train_ds, val_fraction, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_cfg)
    # drop_last=False on validation: a small held-out set with drop_last=True
    # silently yields zero batches and reports no metrics at all.
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **{**loader_cfg, "drop_last": False})
        if val_ds is not None
        else None
    )
    return train_loader, val_loader
