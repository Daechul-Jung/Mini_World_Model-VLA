"""Open X-Embodiment pick-and-place subset -> training windows.

Reads the `.npz` episodes written by `scripts/download/download_openx_subset.py`:

    images       (T, H, W, 3) uint8
    actions      (T, A) float32
    rewards      (T,) float64
    instruction  scalar string

Yields one window per sample:

    image        (obs_horizon, 3, H, W)  float in [-1, 1]
    actions      (action_chunk, A)       normalised to [-1, 1]
    instruction  str
    pad_mask     (obs_horizon,)          False where history was padded

**Action normalisation is not optional.** Raw OpenX actions are physical
quantities with wildly different scales per dimension (metres of translation vs
radians of rotation vs a binary gripper). Training on raw values means the loss
is dominated by whichever dimension happens to have the largest units. The
`ActionSpec` computed here (q01/q99 per dimension, matching OpenVLA's convention)
is saved into the checkpoint, because a policy without its normalisation
statistics cannot be deployed -- the numbers it emits have no units.

**The gripper dimension is special.** It is binary, so percentile normalisation
would map it to the extremes and the 1st/99th percentile would be identical.
It is excluded from percentile scaling and passed through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common.types import ActionSpec
from vla.core.registry import VLA_DATASETS


@VLA_DATASETS.register("openx_npz", note="pick-and-place subset from download_openx_subset.py")
class OpenXWindowDataset(Dataset):
    """Sliding windows over OpenX episodes.

    Args:
        root: directory of `episode_*.npz` plus `metadata.json`.
        obs_horizon: frames of history per sample (must match `PolicySpec`).
        action_chunk: actions predicted per sample.
        instruction_filter: substring match on the instruction, e.g. `"pick"`.
            The stated goal is a pick-and-place subset; this is where that filter
            is applied, and it also documents in the config which episodes a
            checkpoint was trained on.
        gripper_index: which action dim is the binary gripper (`None` if there
            is none). UCSD pick-place is 4-DoF with the gripper last.
    """

    def __init__(
        self,
        root: str = "data/openx",
        image_size: int = 128,
        obs_horizon: int = 2,
        action_chunk: int = 4,
        stride: int = 1,
        instruction_filter: Optional[str] = None,
        gripper_index: Optional[int] = -1,
        action_spec: Optional[ActionSpec] = None,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.obs_horizon = obs_horizon
        self.action_chunk = action_chunk
        self.gripper_index = gripper_index
        self.augment = augment

        self.episodes = self._load_metadata(instruction_filter)
        if not self.episodes:
            raise FileNotFoundError(
                f"no episodes in {root}" + (f" matching {instruction_filter!r}" if instruction_filter else "")
            )

        self.index: List[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            length = ep["length"]
            # A window needs obs_horizon-1 frames of history and action_chunk
            # actions ahead, so valid centres are bounded on both sides.
            for t in range(0, max(length - action_chunk, 0) + 1, stride):
                self.index.append((ei, t))

        self.action_spec = action_spec or self._compute_action_spec()

    # ---------------------------------------------------------------- metadata

    def _load_metadata(self, instruction_filter: Optional[str]) -> List[Dict[str, Any]]:
        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
        else:
            raw = [{"path": str(p)} for p in sorted(self.root.glob("episode_*.npz"))]

        episodes = []
        for entry in raw:
            path = Path(entry["path"])
            if not path.is_absolute() and not path.exists():
                path = self.root.parent / entry["path"]
            if not path.exists():
                path = self.root / Path(entry["path"]).name
            if not path.exists():
                continue
            instruction = entry.get("instruction", "")
            if instruction_filter and instruction_filter.lower() not in instruction.lower():
                continue
            with np.load(path, allow_pickle=True) as d:
                length = int(d["images"].shape[0])
                if not instruction:
                    instruction = str(d["instruction"]) if "instruction" in d else ""
            episodes.append({"path": path, "instruction": instruction, "length": length})
        return episodes

    def _compute_action_spec(self, max_episodes: int = 200) -> ActionSpec:
        """Percentile statistics over the filtered episodes.

        Computed once at construction and cached in the checkpoint. Recomputing
        on a different subset silently changes what a saved policy's outputs
        mean, so the spec travels with the weights.
        """
        chunks = []
        for ep in self.episodes[:max_episodes]:
            with np.load(ep["path"], allow_pickle=True) as d:
                chunks.append(np.asarray(d["actions"], dtype=np.float32))
        actions = np.concatenate(chunks, axis=0)

        q01 = torch.from_numpy(np.percentile(actions, 1, axis=0).astype(np.float32))
        q99 = torch.from_numpy(np.percentile(actions, 99, axis=0).astype(np.float32))
        if self.gripper_index is not None:
            # Binary dim: pass through untouched so [-1, 1] scaling is a no-op.
            q01[self.gripper_index] = -1.0
            q99[self.gripper_index] = 1.0
        # Guard against a constant dimension collapsing the range to zero.
        degenerate = (q99 - q01).abs() < 1e-6
        q01[degenerate], q99[degenerate] = -1.0, 1.0

        return ActionSpec(
            dim=actions.shape[1],
            mean=torch.from_numpy(actions.mean(0)),
            std=torch.from_numpy(actions.std(0)),
            q01=q01,
            q99=q99,
            gripper_index=self.gripper_index,
            name=f"openx:{self.root.name}",
        )

    # ------------------------------------------------------------------- items

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ei, t = self.index[idx]
        ep = self.episodes[ei]

        with np.load(ep["path"], allow_pickle=True) as d:
            images, actions = d["images"], d["actions"]
            length = images.shape[0]

            hist = [max(t - i, 0) for i in reversed(range(self.obs_horizon))]
            frames = torch.from_numpy(images[hist].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1.0

            future = [min(t + i, length - 1) for i in range(self.action_chunk)]
            chunk = torch.from_numpy(actions[future].copy()).float()

        if frames.shape[-1] != self.image_size:
            frames = torch.nn.functional.interpolate(
                frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        if self.augment:
            frames = self._augment(frames)

        return {
            "image": frames,
            "actions": self.action_spec.normalize(chunk),
            "instruction": ep["instruction"],
            # False where history was clamped to the episode start.
            "pad_mask": torch.tensor([h > 0 or t == 0 for h in hist], dtype=torch.bool),
        }

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Colour jitter + small random shift, applied identically across time.

        Per-frame augmentation would inject fake motion, which is exactly the
        signal an obs_horizon>1 policy is trying to read.
        """
        if torch.rand(1) < 0.5:
            frames = frames * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.1
        shift = int(self.image_size * 0.04)
        if shift > 0:
            dx, dy = torch.randint(-shift, shift + 1, (2,)).tolist()
            frames = torch.roll(frames, shifts=(dy, dx), dims=(-2, -1))
        return frames.clamp(-1, 1)


def collate_windows(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that keeps `instruction` a list of strings."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "pad_mask": torch.stack([b["pad_mask"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }
