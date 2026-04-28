# Mini World Model + VLA

A personal research project building:
1. **VLA** — Vision-Language-Action model (PyTorch port of [Octo](https://octo-models.github.io/))
2. **Generative World Model** — Genie/DIAMOND-style video prediction with diffusion decoding
3. **Image Diffusion** — Latent Diffusion Model (LDM) for room image generation
4. **3D Gaussian Splatting** — 3D room reconstruction from personal or TUM RGB-D video

The long-term goal is to train VLA and RL agents *inside* the world model as a generative simulation environment, avoiding costly real-world data collection.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GenieWorldModel                          │
│                                                             │
│  Video ──► VQ-VAE ──► discrete tokens                       │
│            tokenizer   (B, T, h*w)                          │
│                              │                              │
│                         DynamicsTransformer                 │
│                         (causal GPT-style)                  │
│                              │                              │
│                       next frame tokens                     │
│                              │                              │
│                         Diffusion UNet ──► imagined frame   │
│                         (DDPM/DDIM)                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            LatentDiffusionModel (LDM)                       │
│                                                             │
│  Image ──► VAE encoder ──► latent z ──► UNet denoiser ──►  │
│            (8× downsample)   (32×32×4)   (cross-attn cond) │
│                              ◄─── trained with DDPM loss    │
│  z ──► VAE decoder ──► generated room image                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   OctoTransformer (VLA)                     │
│                                                             │
│  [task tokens | obs_t=0 | obs_t=1 | ... | readout tokens]  │
│         │           │                          │            │
│   LanguageTokenizer  ImageTokenizer      ActionHead         │
│   (prefix: fixed)   (per timestep)    (diffusion-based)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
Mini_World_Model-VLA/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── world_model/
│   │   └── genie_medium.yaml       # World model hyperparameters
│   └── diffusion/
│       └── ldm_medium.yaml         # LDM hyperparameters
│
├── src/
│   │
│   ├── world_model/                # Generative world model
│   │   ├── tokenizer/
│   │   │   ├── quantizer.py        # VectorQuantizer (VQ straight-through)
│   │   │   └── vqvae.py            # VQ-VAE encoder / decoder
│   │   ├── dynamics/
│   │   │   └── transformer.py      # Causal dynamics transformer (Genie-style)
│   │   ├── decoder/
│   │   │   ├── ddpm.py             # DDPM + DDIM noise scheduler
│   │   │   └── unet.py             # Conditioned UNet denoiser
│   │   ├── splatting/
│   │   │   └── gaussians.py        # 3D Gaussian Splatting representation
│   │   └── world_model.py          # GenieWorldModel: wraps all components
│   │
│   ├── diffusion/                  # Standalone image diffusion (LDM)
│   │   ├── vae/
│   │   │   └── autoencoder.py      # VAE encoder / decoder
│   │   ├── schedulers/
│   │   │   ├── ddpm.py             # DDPM noise scheduler
│   │   │   └── ddim.py             # DDIM fast sampler
│   │   └── ldm.py                  # LatentDiffusionModel (VAE + UNet)
│   │
│   ├── vla/                        # VLA model (Octo-based, do not edit)
│   │   ├── model/
│   │   │   ├── components/
│   │   │   │   ├── base.py         # TokenGroup dataclass
│   │   │   │   ├── block_transformer.py  # BlockTransformer + attention rules
│   │   │   │   ├── transformer.py  # Transformer, MAPHead
│   │   │   │   ├── tokenizers.py   # ImageTokenizer, LanguageTokenizer, etc.
│   │   │   │   ├── action_heads.py # Action prediction heads
│   │   │   │   ├── diffusion.py    # Diffusion action head
│   │   │   │   └── vit_encoder.py  # ViT visual encoder
│   │   │   ├── octo_module.py      # OctoTransformer + OctoModule
│   │   │   └── octo_model.py       # Full model + training utilities
│   │   ├── data/                   # Data loading for VLA
│   │   ├── utils/
│   │   │   ├── spec.py             # ModuleSpec for config-driven construction
│   │   │   └── typing.py
│   │   └── scripts/                # VLA training scripts
│   │
│   └── world/                      # (legacy stubs — see src/world_model/)
│
├── scripts/
│   ├── download/
│   │   ├── download_openx_subset.py   # Bridge dataset (pick & place)
│   │   ├── download_lsun_rooms.py     # LSUN bedroom / living_room
│   │   └── download_tum_rgbd.py       # TUM RGB-D indoor sequences
│   └── train/
│       ├── train_world_model.py       # 3-phase world model training
│       └── train_diffusion.py         # 2-phase LDM training
│
├── notebooks/                         # Exploration notebooks
│
└── checkpoints/                       # Saved weights (auto-created)
    ├── world_model/
    └── diffusion/
```

---

## Model sizes (personal GPU friendly)

| Component | Params | VRAM (bf16) |
|---|---|---|
| VQ-VAE | ~40M | ~2 GB |
| Dynamics Transformer | ~85M | ~4 GB |
| Diffusion UNet (decoder) | ~150M | ~6 GB |
| **World Model total** | **~275M** | **~12 GB** |
| VAE (for LDM) | ~84M | ~3 GB |
| LDM UNet | ~116M | ~5 GB |
| **LDM total** | **~200M** | **~8 GB** |

Use `torch.autocast("cuda", dtype=torch.bfloat16)` and `batch_size=8–16` on a 16–24 GB GPU.

---

## Quickstart

### 1. Download data

```bash
# TUM RGB-D (room sequences, ~150 MB total)
python scripts/download/download_tum_rgbd.py \
    --sequences fr1_desk fr2_desk fr3_office \
    --output_dir data/tum_rgbd

# Bridge dataset subset for VLA (pick & place, ~5 GB)
python scripts/download/download_openx_subset.py \
    --n_episodes 500 --output_dir data/openx

# LSUN room images for image diffusion (~20K images)
python scripts/download/download_lsun_rooms.py \
    --n_images 20000 --output_dir data/lsun_rooms
```

### 2. Train world model (3 phases)

```bash
# Phase 1: VQ-VAE (tokenize frames)
python scripts/train/train_world_model.py \
    --data_dir data/tum_rgbd --phase 1 --epochs 50 --batch_size 16

# Phase 2: Dynamics Transformer (predict next tokens)
python scripts/train/train_world_model.py \
    --data_dir data/tum_rgbd --phase 2 --epochs 100 \
    --vqvae_ckpt checkpoints/world_model/vqvae_best.pt

# Phase 3: Diffusion decoder (render imagined frames)
python scripts/train/train_world_model.py \
    --data_dir data/tum_rgbd --phase 3 --epochs 100 \
    --vqvae_ckpt checkpoints/world_model/vqvae_best.pt
```

### 3. Train image diffusion (rooms)

```bash
python scripts/train/train_diffusion.py \
    --data_dir data/lsun_rooms --phase all --epochs 100 --batch_size 32
```

### 4. Generate imagined room sequences

```python
from src.world_model import GenieWorldModel
import torch

model = GenieWorldModel.create_medium()
model.load_state_dict(...)   # load your checkpoints

context = torch.randn(1, 4, 3, 256, 256)   # 4 observed frames
imagined = model.imagine(context, n_steps=8, ddim_steps=50)
# → (1, 8, 3, 256, 256) generated frames
```

---

## Using personal room video

**Yes, personal video works perfectly** for both world model training and 3DGS reconstruction.

**For world model video training:**
1. Record a slow walkthrough of your room (phone/camera, good lighting, 1–5 minutes).
2. Extract frames: `ffmpeg -i room.mp4 -vf fps=10 data/myroom/frame_%05d.png`
3. Run training: `python scripts/train/train_world_model.py --data_dir data/myroom`

**For 3D Gaussian Splatting (3D room reconstruction):**
1. Record video moving slowly around the room (keep features visible at all times).
2. Estimate camera poses with COLMAP:
   ```bash
   pip install pycolmap
   colmap automatic_reconstructor --image_path data/myroom --workspace_path data/colmap_out
   ```
3. Initialize Gaussians from the COLMAP sparse point cloud:
   ```python
   from src.world_model.splatting.gaussians import GaussianScene
   scene = GaussianScene.from_colmap_points(xyz, rgb)
   ```
4. Render with the `gsplat` library: `pip install gsplat`

**TUM RGB-D shortcut:** Already has calibrated depth + ground-truth poses, so skip COLMAP entirely. Download and run directly with `download_tum_rgbd.py`.

---

## Key design decisions

**Why VQ-VAE + Transformer (not end-to-end pixel diffusion)?**
Discretizing frames into tokens (like VQVAE → codebook) lets the dynamics model be a simple language-model-style GPT, which is much cheaper to train than full pixel-space diffusion. The separate diffusion decoder handles image quality. This is the Genie/IRIS architecture.

**Why 3-phase training?**
Each component has a different objective. Joint end-to-end training is unstable and requires careful loss balancing. Training separately then fine-tuning jointly (optional) gives more reliable results.

**Why LDM instead of pixel diffusion (DDPM on raw images)?**
Pixel diffusion on 256×256 is extremely compute-intensive (~300× more UNet evaluations than latent diffusion). LDM (VAE + UNet in latent space) achieves comparable quality in 8× less memory/compute.

---

## References

- Octo: [octo-models.github.io](https://octo-models.github.io/)
- Genie: Bruce et al., 2024 — Generative Interactive Environments
- DIAMOND: Micheli et al., 2024 — Diffusion world model
- VQ-VAE: van den Oord et al., 2017
- LDM: Rombach et al., CVPR 2022 (Stable Diffusion paper)
- 3DGS: Kerbl et al., SIGGRAPH 2023
- TUM RGB-D: Sturm et al., IROS 2012
- Open X-Embodiment: Padalkar et al., 2023
