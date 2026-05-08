# Mini World Model + VLA

Personal research code for studying diffusion models, Octo-style VLA models, and small generative world models for rooms.

The current direction is:

1. **Image diffusion**: latent diffusion for room image generation.
2. **World model**: video-token world model that predicts future room frames, with a diffusion decoder for pixels.
3. **3D room model**: Gaussian Splatting from TUM RGB-D or personal room video.
4. **VLA**: PyTorch Octo-style model trained first on a small Open X-Embodiment task subset.

The VLA/Octo implementation lives under `src/vla/` and is intentionally separate while it is being studied.

## Main Ideas

### Diffusion Model

For image generation, use a latent diffusion model:

```text
room image -> VAE latent -> UNet denoising diffusion -> generated latent -> VAE decoder -> image
```

Good starter data:

- **LSUN bedroom/living_room/kitchen**: good for room images, but full categories are large.
- **ADE20K / Places-style indoor images**: useful smaller indoor-scene alternatives.
- **ImageNet subset**: useful for learning image generation mechanics, but not ideal for room geometry.
- **Personal room photos/video frames**: best for generating your own room style.

### World Model

The world model is trained from **videos or ordered image sequences**, not isolated 2D images. ImageNet can help train an image prior, but it is not enough by itself to learn a navigable 3D world because it has no camera motion, temporal continuity, depth, or action labels.

Current world model:

```text
video frames -> VQ-VAE tokenizer -> discrete frame tokens
tokens/actions -> causal dynamics transformer -> next-frame tokens
next-frame tokens -> diffusion decoder -> generated future frame
```

For better 3D room results, add geometry:

```text
RGB-D frames + poses -> 3D Gaussian Splatting -> render novel views
video tokens + diffusion -> predict/generate future appearance
```

### 3D Gaussian Splatting

The project now includes a small `gsplat` path:

- `src/world_model/splatting/gaussians.py`: learnable Gaussian scene and renderer wrapper.
- `scripts/train/train_gaussian_splatting.py`: small TUM RGB-D trainer.

TUM RGB-D is easier than personal video because it includes RGB, depth, camera intrinsics, and ground-truth camera poses. Personal video can work, but you need camera poses from COLMAP/SLAM, and monocular video has less reliable scale/depth than RGB-D.

## Project Structure

```text
Mini_World_Model-VLA/
├── README.md
├── requirements.txt
├── requirements-openx.txt
├── configs/
│   ├── diffusion/ldm_medium.yaml
│   ├── vla_configs.yaml
│   └── world_model/genie_medium.yaml
├── scripts/
│   ├── download/
│   │   ├── download_tum_rgbd.py
│   │   ├── download_lsun_rooms.py
│   │   ├── download_openx_subset.py
│   │   └── prepare_imagenet_subset.py
│   └── train/
│       ├── train_diffusion.py
│       ├── train_world_model.py
│       └── train_gaussian_splatting.py
├── src/
│   ├── diffusion/
│   │   ├── ldm.py
│   │   ├── schedulers/
│   │   └── vae/
│   ├── world_model/
│   │   ├── world_model.py
│   │   ├── tokenizer/
│   │   ├── dynamics/
│   │   ├── decoder/
│   │   └── splatting/
│   ├── dataset/
│   └── vla/
└── data/
    ├── tum_rgbd/
    ├── lsun_rooms/
    ├── openx/
    └── imagenet_subset/
```

## Install

Use your normal PyTorch environment for diffusion, world model, and 3DGS:

```bash
python3 -m pip install -r requirements.txt
```

Use a separate environment for OpenX/TFDS downloads:

```bash
# On Ubuntu/Debian, install this first if venv creation has no pip:
# sudo apt install python3.10-venv
python3 -m venv venv_openx_download
./venv_openx_download/bin/python -m pip install -r requirements-openx.txt
```


```bash
which python3
python3 -m pip --version
python3 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

In this workspace, `venv_openx_download` existed but had no `pip`, and `ensurepip` is unavailable because the system Python is missing the Debian/Ubuntu `python3.10-venv` package. Install that package, recreate the venv, then install `requirements-openx.txt`.

## Download Data

### TUM RGB-D for world model and 3DGS

You already installed `fr1_desk` and `fr2_desk`, so the default command does not include `fr3_desk` or `fr3_office`.

```bash
python3 scripts/download/download_tum_rgbd.py \
    --sequences fr1_desk fr2_desk \
    --output_dir data/tum_rgbd
```

This exports:

```text
data/tum_rgbd/fr1_desk/rgb_frames/
data/tum_rgbd/fr1_desk/depth_frames/
data/tum_rgbd/fr1_desk/poses.txt
data/tum_rgbd/fr1_desk/intrinsics.json
```

### OpenX subset for VLA

Start with UCSD pick-and-place:

```bash
./venv_openx_download/bin/python scripts/download/download_openx_subset.py \
    --dataset ucsd_pick_place \
    --n_episodes 100 \
    --output_dir data/openx
```

Use `--n_episodes 500` later if the first run works and storage is okay.

### LSUN rooms for image diffusion

```bash
python3 scripts/download/download_lsun_rooms.py \
    --categories bedroom living_room kitchen \
    --n_images 5000 \
    --output_dir data/lsun_rooms
```

LSUN full downloads are large. For a personal GPU, 5k-20k images is enough to study the pipeline.

### ImageNet subset

ImageNet usually requires manually accepting terms and downloading from the official source. After extracting ImageNet train folders:

```bash
python3 scripts/download/prepare_imagenet_subset.py \
    --imagenet_root /path/to/imagenet/train \
    --output_dir data/imagenet_subset \
    --classes 50 \
    --images_per_class 100
```

## Train

### Latent diffusion for room images

```bash
python3 scripts/train/train_diffusion.py \
    --data_dir data/lsun_rooms \
    --phase all \
    --epochs 50 \
    --batch_size 16
```

### Tokenized video world model

```bash
python3 scripts/train/train_world_model.py \
    --data_dir data/tum_rgbd \
    --phase all \
    --size small \
    --epochs 20 \
    --batch_size 8
```

Use `--size medium` after the small path is working.

### 3D Gaussian Splatting room model

```bash
python3 scripts/train/train_gaussian_splatting.py \
    --sequence_dir data/tum_rgbd/fr1_desk \
    --steps 1000 \
    --max_points 50000 \
    --image_size 256
```

This is a learning implementation, not a full production 3DGS trainer. It is useful for understanding the components and getting a small room reconstruction loop running.
Training needs a CUDA GPU because `gsplat` rasterization is GPU-oriented.

## Personal Room Video

Personal video is a good next step. For the frame-prediction world model:

```bash
mkdir -p data/myroom/rgb_frames
ffmpeg -i room.mp4 -vf fps=10 data/myroom/rgb_frames/%05d.png
python3 scripts/train/train_world_model.py --data_dir data/myroom --phase all --size small
```

For 3DGS from personal video, first estimate camera poses and sparse points with COLMAP or another SLAM system, then initialize Gaussian Splatting from that geometry. If you record RGB-D video, the path is much easier because depth gives direct 3D points.

## Practical Training Order

1. Train/test LDM on a tiny LSUN or personal-room image subset.
2. Train/test world-model VQ-VAE on TUM `fr1_desk`.
3. Train the dynamics transformer on short clips from `fr1_desk` and `fr2_desk`.
4. Train the diffusion decoder for sharper imagined frames.
5. Train 3DGS on `fr1_desk` to get novel-view room rendering.
6. Train VLA on `ucsd_pick_place` subset.
7. Later connect VLA/RL rollouts to generated frames or rendered 3DGS views.

## References

- Octo: https://octo-models.github.io/
- Open X-Embodiment / RT-X: https://robotics-transformer-x.github.io/
- TensorFlow Datasets catalog: https://www.tensorflow.org/datasets/catalog/overview
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- LSUN: https://www.yf.io/p/lsun
- Latent Diffusion Models: https://arxiv.org/abs/2112.10752
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Genie: https://arxiv.org/abs/2402.15391
