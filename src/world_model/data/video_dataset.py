"""Ordered-clip datasets for stages B, C and D.

Two sources, deliberately kept behind one output format:

* `VideoFolderDataset` -- a directory of ordered frames (TUM RGB-D `rgb_frames/`,
  or `ffmpeg`-extracted frames from a phone video). Optionally emits the depth
  and pose supervision TUM ships, for the `physics/` auxiliary heads.
* `EpisodeNPZDataset` -- the OpenX `.npz` episodes in `data/openx/`, which carry
  real robot actions. This is the only dataset here with *labelled* actions, so
  it is what an action-conditioned (`action_kind="robot"`) dynamics model trains
  on, and what makes the world model usable as a VLA environment.

Both yield:
    {"frames": (T, 3, H, W) in [-1, 1],
     "actions": (T-1, A) float          -- EpisodeNPZDataset only
     "depth":   (T, 1, H, W)            -- VideoFolderDataset with depth enabled
     "pose":    (T, 7)                  -- VideoFolderDataset with poses enabled
     "instruction": str}                -- EpisodeNPZDataset only

A clip is a contiguous window of `clip_len` frames sampled with `stride`. Frame
*rate* matters more than resolution for a world model: two frames 1/30 s apart
differ by almost nothing, and a latent action model trained on them learns that
the action is always "no-op". Genie trains at 10 fps; subsample accordingly with
`frame_skip`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from world_model.core.registry import WM_DATASETS

from .image_dataset import IMAGE_SUFFIXES


def _load_frame(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    # np.array (not np.asarray): a PIL image exposes a read-only buffer, and
    # torch.from_numpy on it warns about non-writable memory on every worker.
    # np.array copies, which the subsequent .float() would do anyway.
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return arr


@WM_DATASETS.register("video_folder", note="ordered frames; TUM RGB-D or extracted video")
class VideoFolderDataset(Dataset):
    """Sliding-window clips over one or more ordered-frame directories.

    Args:
        roots: directories, each holding one sequence's frames in sort order. A
            TUM sequence root works directly (`rgb_frames/` is found by name).
        clip_len: frames per clip. Stage B needs >= 2; stage C wants >= 8.
        stride: step between consecutive clip start indices.
        frame_skip: take every Nth frame *within* a clip. Use this to hit ~10 fps
            from a 30 fps source rather than resampling files on disk.
        with_depth / with_pose: emit TUM's depth maps and ground-truth poses for
            the `physics/` auxiliary heads. Silently disabled if absent.
    """

    def __init__(
        self,
        roots: str | Sequence[str],
        image_size: int = 128,
        clip_len: int = 8,
        stride: int = 4,
        frame_skip: int = 1,
        with_depth: bool = False,
        with_pose: bool = False,
        frame_subdir: str = "rgb_frames",
    ):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.image_size = image_size
        self.clip_len = clip_len
        self.frame_skip = frame_skip
        self.with_depth = with_depth
        self.with_pose = with_pose

        self.sequences: List[Dict[str, Any]] = []
        for root in roots:
            for seq_dir in self._find_sequences(Path(root), frame_subdir):
                frames = sorted(
                    p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                )
                span = clip_len * frame_skip
                if len(frames) < span:
                    continue
                self.sequences.append(
                    {
                        "dir": seq_dir,
                        "frames": frames,
                        "depth": self._sibling(seq_dir, "depth_frames") if with_depth else None,
                        "poses": self._load_poses(seq_dir) if with_pose else None,
                    }
                )

        self.index: List[Tuple[int, int]] = []
        for si, seq in enumerate(self.sequences):
            span = clip_len * frame_skip
            for start in range(0, len(seq["frames"]) - span + 1, stride):
                self.index.append((si, start))

        if not self.index:
            raise FileNotFoundError(
                f"no clips of length {clip_len} (skip {frame_skip}) found under {list(roots)}"
            )

    @staticmethod
    def _find_sequences(root: Path, frame_subdir: str) -> List[Path]:
        """Accept a frames dir, a TUM sequence dir, or a parent of several."""
        if (root / frame_subdir).is_dir():
            return [root / frame_subdir]
        nested = sorted(p / frame_subdir for p in root.iterdir() if (p / frame_subdir).is_dir())
        if nested:
            return nested
        return [root] if any(p.suffix.lower() in IMAGE_SUFFIXES for p in root.iterdir()) else []

    @staticmethod
    def _sibling(seq_dir: Path, name: str) -> Optional[Path]:
        cand = seq_dir.parent / name
        return cand if cand.is_dir() else None

    @staticmethod
    def _load_poses(seq_dir: Path) -> Optional[np.ndarray]:
        """TUM `poses.txt`: `timestamp tx ty tz qx qy qz qw` per line."""
        path = seq_dir.parent / "poses.txt"
        if not path.exists():
            return None
        rows = [
            [float(v) for v in line.split()[1:8]]
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return np.asarray(rows, dtype=np.float32) if rows else None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, start = self.index[idx]
        seq = self.sequences[si]
        picks = [start + i * self.frame_skip for i in range(self.clip_len)]

        out: Dict[str, Any] = {
            "frames": torch.stack([_load_frame(seq["frames"][i], self.image_size) for i in picks])
        }
        if seq.get("depth"):
            depths = sorted(seq["depth"].iterdir())
            if len(depths) > picks[-1]:
                maps = [
                    torch.from_numpy(
                        np.asarray(
                            Image.open(depths[i]).resize(
                                (self.image_size, self.image_size), Image.NEAREST
                            )
                        ).astype(np.float32)
                    )
                    for i in picks
                ]
                # TUM depth PNGs are uint16 millimetres scaled by 5000.
                out["depth"] = torch.stack(maps).unsqueeze(1) / 5000.0
        if seq.get("poses") is not None and len(seq["poses"]) > picks[-1]:
            out["pose"] = torch.from_numpy(seq["poses"][picks])
        return out


@WM_DATASETS.register("episode_npz", note="OpenX episodes; the only source with real actions")
class EpisodeNPZDataset(Dataset):
    """Clips from `data/openx/episode_*.npz` -- frames *and* labelled actions.

    Each `.npz` holds `images (T, H, W, 3) uint8`, `actions (T, A) float32`,
    `rewards (T,)`, and a scalar `instruction` string.

    Action alignment: `actions[t]` is the action executed *at* frame t, producing
    frame t+1. The clip therefore returns `actions[:-1]`, one per transition,
    matching what `Dynamics.forward` expects. Getting this off by one is the
    classic way to train a world model that appears to work and is actually
    predicting the past.
    """

    def __init__(
        self,
        root: str = "data/openx",
        image_size: int = 128,
        clip_len: int = 8,
        stride: int = 4,
        frame_skip: int = 1,
        instruction_filter: Optional[str] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.clip_len = clip_len
        self.frame_skip = frame_skip

        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            episodes = json.loads(meta_path.read_text())
        else:
            episodes = [
                {"path": str(p), "instruction": "", "length": None}
                for p in sorted(self.root.glob("episode_*.npz"))
            ]

        if instruction_filter:
            needle = instruction_filter.lower()
            episodes = [e for e in episodes if needle in e.get("instruction", "").lower()]

        self.episodes = episodes
        self.index: List[Tuple[int, int]] = []
        span = clip_len * frame_skip
        for ei, ep in enumerate(episodes):
            length = ep.get("length") or self._peek_length(ep["path"])
            for start in range(0, max(length - span, 0) + 1, stride):
                self.index.append((ei, start))
        if not self.index:
            raise FileNotFoundError(f"no clips in {root} (filter={instruction_filter!r})")

    @staticmethod
    def _peek_length(path: str) -> int:
        with np.load(path, allow_pickle=True) as d:
            return int(d["images"].shape[0])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ei, start = self.index[idx]
        ep = self.episodes[ei]
        picks = [start + i * self.frame_skip for i in range(self.clip_len)]

        with np.load(self.root.parent / ep["path"] if not Path(ep["path"]).is_absolute() and not Path(ep["path"]).exists() else ep["path"], allow_pickle=True) as d:
            imgs = d["images"][picks]
            actions = d["actions"][picks]
            instruction = str(d["instruction"]) if "instruction" in d else ep.get("instruction", "")

        frames = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if frames.shape[-1] != self.image_size:
            frames = torch.nn.functional.interpolate(
                frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return {
            "frames": frames,
            "actions": torch.from_numpy(actions[:-1]).float(),   # one per transition
            "instruction": instruction,
        }
