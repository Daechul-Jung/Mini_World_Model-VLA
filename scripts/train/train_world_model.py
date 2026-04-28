"""
Training script for GenieWorldModel.

Three-phase training (recommended):
  Phase 1: Train VQ-VAE alone for good frame tokenization.
  Phase 2: Freeze VQ-VAE, train DynamicsTransformer on discrete tokens.
  Phase 3: Freeze VQ-VAE + Dynamics, train diffusion decoder for sharp rendering.

Each phase can be run independently by setting --phase 1/2/3.

Usage:
    # Phase 1: VQ-VAE
    python scripts/train/train_world_model.py \
        --data_dir data/tum_rgbd/fr1_desk/rgb_frames \
        --phase 1 --epochs 50 --batch_size 16

    # Phase 2: Dynamics (requires Phase 1 checkpoint)
    python scripts/train/train_world_model.py \
        --data_dir data/tum_rgbd \
        --phase 2 --epochs 100 \
        --vqvae_ckpt checkpoints/vqvae_best.pt

    # Phase 3: Diffusion decoder
    python scripts/train/train_world_model.py \
        --data_dir data/tum_rgbd \
        --phase 3 --epochs 100 \
        --vqvae_ckpt checkpoints/vqvae_best.pt

    # All phases sequentially
    python scripts/train/train_world_model.py \
        --data_dir data/tum_rgbd --phase all
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Single-frame dataset for VQ-VAE and diffusion decoder training."""

    def __init__(self, root: str, image_size: int = 256):
        self.paths = sorted(
            p for p in Path(root).rglob("*.png")
            if not p.name.startswith("depth")
        ) + sorted(Path(root).rglob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class VideoClipDataset(Dataset):
    """
    Video clip dataset for dynamics model training.

    Splits each image folder sequence into overlapping clips of `clip_len` frames.
    Frames must be sorted by name (e.g. 00000.png, 00001.png, ...).
    """

    def __init__(self, root: str, clip_len: int = 8, stride: int = 4, image_size: int = 256):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_len = clip_len
        self.clips = []

        for seq_dir in Path(root).rglob("rgb_frames"):
            frames = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
            for start in range(0, len(frames) - clip_len, stride):
                self.clips.append(frames[start : start + clip_len])

        if not self.clips:
            # Fallback: treat the root itself as a sequence
            frames = sorted(Path(root).glob("*.png")) + sorted(Path(root).glob("*.jpg"))
            for start in range(0, len(frames) - clip_len, stride):
                self.clips.append(frames[start : start + clip_len])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = [Image.open(p).convert("RGB") for p in self.clips[idx]]
        return torch.stack([self.transform(f) for f in clip])  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def train_vqvae(args, model):
    ds = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.vqvae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log.info(f"Phase 1 — VQ-VAE training: {len(ds)} images, {args.epochs} epochs")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.vqvae.train()
        ep_loss = 0.0

        for batch in loader:
            batch = batch.to(args.device)
            _, loss, metrics = model.vqvae(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vqvae.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | loss={ep_loss:.4f}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            _save(model.vqvae, args.ckpt_dir / "vqvae_best.pt")

    _save(model.vqvae, args.ckpt_dir / "vqvae_final.pt")
    log.info("Phase 1 complete.")


def train_dynamics(args, model):
    if args.vqvae_ckpt:
        model.vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=args.device))
        log.info(f"Loaded VQ-VAE from {args.vqvae_ckpt}")

    model.vqvae.eval()
    for p in model.vqvae.parameters():
        p.requires_grad_(False)

    ds = VideoClipDataset(args.data_dir, clip_len=args.clip_len, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size // args.clip_len or 1,
                        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.dynamics.parameters(), lr=args.lr, weight_decay=1e-4)

    log.info(f"Phase 2 — Dynamics training: {len(ds)} clips, {args.epochs} epochs")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.dynamics.train()
        ep_loss = 0.0

        for clips in loader:
            clips = clips.to(args.device)          # (B, T, 3, H, W)
            B, T, C, H, W = clips.shape

            with torch.no_grad():
                flat = clips.reshape(B * T, C, H, W)
                _, _, indices = model.vqvae.encode(flat)
                h, w = indices.shape[1], indices.shape[2]
                indices = indices.reshape(B, T, h, w)

            loss = model.dynamics_loss(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        ep_loss /= len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | dynamics_loss={ep_loss:.4f}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            _save(model.dynamics, args.ckpt_dir / "dynamics_best.pt")

    _save(model.dynamics, args.ckpt_dir / "dynamics_final.pt")
    log.info("Phase 2 complete.")


def train_decoder(args, model):
    if args.vqvae_ckpt:
        model.vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=args.device))

    model.vqvae.eval()
    for p in model.vqvae.parameters():
        p.requires_grad_(False)

    ds = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    log.info(f"Phase 3 — Diffusion decoder training: {len(ds)} images, {args.epochs} epochs")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.decoder.train()
        ep_loss = 0.0

        for batch in loader:
            batch = batch.to(args.device)

            with torch.no_grad():
                z_q, _, _ = model.vqvae.encode(batch)

            loss = model.diffusion_loss(batch, z_q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        ep_loss /= len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | diffusion_loss={ep_loss:.4f}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            _save(model.decoder, args.ckpt_dir / "decoder_best.pt")

    _save(model.decoder, args.ckpt_dir / "decoder_final.pt")
    log.info("Phase 3 complete.")


def _save(module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    log.info(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--phase", default="1", choices=["1", "2", "3", "all"])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--clip_len", type=int, default=8, help="Frames per clip for dynamics")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints/world_model"))
    p.add_argument("--vqvae_ckpt", default=None, help="Path to pre-trained VQ-VAE weights")
    p.add_argument("--action_dim", type=int, default=0, help="0 = no action conditioning")
    p.add_argument("--size", default="medium", choices=["small", "medium"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.world_model.world_model import GenieWorldModel

    model = (
        GenieWorldModel.create_medium(args.action_dim, args.image_size)
        if args.size == "medium"
        else GenieWorldModel.create_small(args.action_dim, args.image_size)
    )
    model = model.to(args.device)
    log.info(f"Model created ({args.size}), device={args.device}")

    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "1":
            train_vqvae(args, model)
            args.vqvae_ckpt = str(args.ckpt_dir / "vqvae_best.pt")
        elif phase == "2":
            train_dynamics(args, model)
        elif phase == "3":
            train_decoder(args, model)


if __name__ == "__main__":
    main()
