"""
Training script for Latent Diffusion Model (LDM) for room image generation.

Two-phase training:
  Phase 1: Train VAE for image compression.
  Phase 2: Freeze VAE, train UNet in latent space.

Usage:
    # Train VAE on LSUN bedroom images
    python scripts/train/train_diffusion.py \
        --data_dir data/lsun_rooms/bedroom \
        --phase 1 --epochs 50 --batch_size 32

    # Train LDM (UNet) after VAE is ready
    python scripts/train/train_diffusion.py \
        --data_dir data/lsun_rooms \
        --phase 2 --epochs 200 \
        --vae_ckpt checkpoints/diffusion/vae_best.pt

    # Generate images after training
    python scripts/train/train_diffusion.py \
        --generate --n_samples 16 \
        --vae_ckpt checkpoints/diffusion/vae_best.pt \
        --unet_ckpt checkpoints/diffusion/unet_best.pt
"""

import argparse
import logging
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.JPEG")
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(Path(root).rglob(ext)))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def train_vae(args, model):
    ds = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.vae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log.info(f"Phase 1 — VAE: {len(ds)} images, {args.epochs} epochs")
    best = float("inf")

    for epoch in range(args.epochs):
        model.vae.train()
        ep_loss = 0.0

        for batch in loader:
            batch = batch.to(args.device)
            _, loss = model.vae(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | vae_loss={ep_loss:.5f}")

        if ep_loss < best:
            best = ep_loss
            _save(model.vae, args.ckpt_dir / "vae_best.pt")

    _save(model.vae, args.ckpt_dir / "vae_final.pt")
    log.info("Phase 1 complete.")


def train_ldm(args, model):
    if args.vae_ckpt:
        model.vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device))
        log.info(f"Loaded VAE from {args.vae_ckpt}")

    model.vae.eval()
    for p in model.vae.parameters():
        p.requires_grad_(False)

    ds = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log.info(f"Phase 2 — LDM UNet: {len(ds)} images, {args.epochs} epochs")
    best = float("inf")

    for epoch in range(args.epochs):
        model.unet.train()
        ep_loss = 0.0

        for batch in loader:
            batch = batch.to(args.device)
            # Unconditional training: conditioning=None
            loss = model.diffusion_loss(batch, conditioning=None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | ldm_loss={ep_loss:.5f}")

        if ep_loss < best:
            best = ep_loss
            _save(model.unet, args.ckpt_dir / "unet_best.pt")

        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            _generate_samples(model, args, epoch + 1)

    _save(model.unet, args.ckpt_dir / "unet_final.pt")
    log.info("Phase 2 complete.")


@torch.no_grad()
def _generate_samples(model, args, step: int, n: int = 8) -> None:
    model.unet.eval()
    images = model.generate(
        batch_size=n,
        device=args.device,
        ddim_steps=50,
    )
    grid = vutils.make_grid(images * 0.5 + 0.5, nrow=4).clamp(0, 1)
    out = args.ckpt_dir / "samples" / f"step_{step:05d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(out))
    log.info(f"  Saved sample grid → {out}")
    model.unet.train()


def _save(module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    log.info(f"  Saved → {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/lsun_rooms")
    p.add_argument("--phase", default="1", choices=["1", "2", "all"])
    p.add_argument("--generate", action="store_true")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints/diffusion"))
    p.add_argument("--vae_ckpt", default=None)
    p.add_argument("--unet_ckpt", default=None)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--context_dim", type=int, default=512,
                   help="Conditioning embedding dim (512 for unconditional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.diffusion.ldm import LatentDiffusionModel

    model = LatentDiffusionModel.create_medium(context_dim=args.context_dim)
    model = model.to(args.device)
    log.info(f"LDM created, device={args.device}")

    if args.generate:
        if args.vae_ckpt:
            model.vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device))
        if args.unet_ckpt:
            model.unet.load_state_dict(torch.load(args.unet_ckpt, map_location=args.device))
        _generate_samples(model, args, step=0, n=args.n_samples)
        return

    phases = ["1", "2"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "1":
            train_vae(args, model)
            args.vae_ckpt = str(args.ckpt_dir / "vae_best.pt")
        elif phase == "2":
            train_ldm(args, model)


if __name__ == "__main__":
    main()
