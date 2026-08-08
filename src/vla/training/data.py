"""Dataloader construction for VLA stages."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from common.types import ActionSpec
from vla.core.registry import VLA_DATASETS
from vla.data.openx_dataset import collate_windows


def build_vla_loaders(
    cfg: Dict[str, Any], batch_size: int, seed: int = 0
) -> Tuple[DataLoader, Optional[DataLoader], ActionSpec]:
    """Return `(train, val, action_spec)`.

    The split is **by episode**, not by window. Splitting by window would put
    frames from the same trajectory on both sides, and since consecutive frames
    are nearly identical, validation error would report memorisation as
    generalisation.
    """
    data_cfg = dict(cfg["data"])
    num_workers = data_cfg.pop("num_workers", 4)
    val_fraction = data_cfg.pop("val_fraction", 0.1)

    dataset = VLA_DATASETS.build(data_cfg)
    action_spec = dataset.action_spec

    n_episodes = len(dataset.episodes)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_episodes, generator=generator).tolist()
    n_val = max(int(n_episodes * val_fraction), 1) if n_episodes > 4 else 0
    val_episodes = set(order[:n_val])

    train_idx = [i for i, (ei, _) in enumerate(dataset.index) if ei not in val_episodes]
    val_idx = [i for i, (ei, _) in enumerate(dataset.index) if ei in val_episodes]

    loader_kwargs = dict(
        num_workers=num_workers, pin_memory=True, collate_fn=collate_windows, drop_last=True
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    # drop_last=False: a small held-out set otherwise yields zero batches.
    val_loader = (
        DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
            **{**loader_kwargs, "drop_last": False},
        )
        if val_idx
        else None
    )
    return train_loader, val_loader, action_spec
