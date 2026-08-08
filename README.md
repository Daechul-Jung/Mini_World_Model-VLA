# Mini World Model + VLA

Personal research code for two things and the bridge between them:

1. **A vision-language-action policy** — an Octo-style transformer trained from
   scratch on an Open X-Embodiment pick-and-place subset, plus large pretrained
   backbones (OpenVLA, π0) loaded frozen with new trainable layers on top.
2. **A Genie-style generative world model** — video tokenizer → latent action
   model → action-conditioned dynamics → optional diffusion decoder, trained one
   component at a time.
3. **The bridge** — using the world model as an RL environment to post-train the
   VLA.

Everything is built around one property: **a new idea should be a new file plus a
config line, never a fork of a training loop.**

Runs on a single RTX 4070 Laptop (7.7 GiB).

---

## Design

Every component category has an abstract contract, a registry, and one file per
implementation. Configs refer to components by name.

**VLA** — `backbone → modules → head`:

```
Observation ──▶ [ backbone ] ──▶ features ──▶ [ modules ] ──▶ [ head ] ──▶ Action
                 frozen or         (B,T,D)     your new         action
                 trained                       layers           parameterisation
```

**World model** — four components, four training stages, four checkpoints:

```
frames ──▶ [ tokenizer ] ──▶ tokens ─┐
                                     ├──▶ [ dynamics ] ──▶ next tokens ──▶ [ decoder ] ──▶ pixels
frames ──▶ [ latent action ] ──▶ a_t ┘
```

The architecture follows the **published Genie 1 paper** (arXiv:2402.15391).
Genie 3 has no architecture paper — only a capability blog post — so its stated
capabilities (720p, 24 fps, minute-long consistency) are treated as targets
rather than as a spec.

List everything currently registered:

```bash
python scripts/tools/list_components.py
```

---

## Layout

```
src/
├── common/            registry, config, checkpoint lineage, stage runner, trainer
├── vla/               core / backbones / modules / heads / data / training / rl / eval
├── world_model/       core / tokenizer / latent_action / dynamics / decoder
│                      / memory / physics / data / training / eval
├── bridge/            action translation, world-model env, reward models
└── diffusion/         standalone latent-diffusion study (image prior)

configs/               one YAML per model size and data source
scripts/               download / train / eval / tools
tests/                 contract tests — CPU, seconds
```

---

## Install

```bash
python3 -m pip install -r requirements.txt          # core: torch, torchvision, numpy, pyyaml
python3 -m pip install -r requirements-vla.txt      # optional: OpenVLA / π0 (large downloads)
python3 -m pip install -r requirements-sim.txt      # optional: simulation for evaluation
```

Open X-Embodiment downloads need a separate environment, because TFDS pins
NumPy < 2:

```bash
python3 -m venv venv_openx_download
./venv_openx_download/bin/python -m pip install -r requirements-openx.txt
```

---

## Data

```bash
# Robot episodes for the VLA (and for an action-conditioned world model)
./venv_openx_download/bin/python scripts/download/download_openx_subset.py \
    --dataset ucsd_pick_place --n_episodes 100 --output_dir data/openx

# Video with camera motion, depth and ground-truth poses
python3 scripts/download/download_tum_rgbd.py --sequences fr1_desk fr2_desk \
    --output_dir data/tum_rgbd

# Room stills — tokenizer training only
python3 scripts/download/download_lsun_rooms.py --categories bedroom \
    --n_images 5000 --output_dir data/lsun_rooms
```

What each source can train:

| Source | Stage A (tokenizer) | Stage B (latent action) | Stage C (dynamics) |
|--------|:---:|:---:|:---:|
| LSUN rooms (stills) | yes | **no** | **no** |
| TUM RGB-D (video) | yes | yes | yes |
| OpenX episodes (video + actions) | yes | yes | yes |

LSUN is an unordered collection of photographs of *different* rooms. There is no
next frame, so it cannot train anything temporal.

---

## Train

Components are trained one at a time; each stage loads its predecessors frozen
and writes its own checkpoint.

```bash
# World model
python scripts/train/train_world_model.py --stage a --config genie_small_lsun.yaml
python scripts/train/train_world_model.py --stage b --config genie_small.yaml
python scripts/train/train_world_model.py --stage c --config genie_small.yaml \
    --tokenizer_ckpt stage_a_tokenizer:best \
    --latent_action_ckpt stage_b_latent_action:best
python scripts/train/train_world_model.py --stage d --config genie_small.yaml \
    --tokenizer_ckpt stage_a_tokenizer:best

# VLA
python scripts/train/train_vla.py --stage bc --config octo_small.yaml
python scripts/train/train_vla.py --stage bc --config openvla_frozen_adapter.yaml
```

`stage_a_tokenizer:best` resolves to the newest run's best checkpoint. Override
any config value inline — values parse as YAML, so lists and dicts work:

```bash
python scripts/train/train_vla.py --stage bc --config octo_small.yaml \
    --set optim.lr=1e-4 \
    --set 'policy.modules=[{"name": "gated_residual", "num_heads": 6}]'
```

Checkpoints land in `checkpoints/<project>/<stage>/<run>/` with `config.yaml`,
`manifest.json`, `metrics.jsonl`, `best.pt` and `last.pt`. Each records the
checkpoints it was built from:

```bash
python scripts/tools/inspect_checkpoint.py stage_c_dynamics:best
```

Check a config's memory cost before starting a long run:

```bash
python scripts/tools/vram_probe.py --project vla --config octo_medium.yaml
```

---

## Test

```bash
python -m pytest tests/ -q
```

These are contract tests, not accuracy tests — they check that every registered
component honours its interface, which is what makes swapping components safe.
They run on CPU in seconds.

---

## Current status

| | Status |
|--|--------|
| World model stages A–D | train end-to-end, checkpoints chain correctly |
| VLA behaviour cloning | trains on the OpenX subset |
| Pretrained backbones (OpenVLA, π0) | implemented, not yet run against downloaded weights |
| RL post-training | contracts only |
| **Simulator evaluation** | **not built — the largest gap** |

Without a simulator there is no success rate, and offline action error correlates
only weakly with task success. That is the next piece of work.

---

## References

- Genie: Generative Interactive Environments — https://arxiv.org/abs/2402.15391
- Genie 3 (capabilities, no architecture) — https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- Octo — https://octo-models.github.io/
- OpenVLA — https://arxiv.org/abs/2406.09246
- π0 — https://arxiv.org/abs/2410.24164
- Open X-Embodiment / RT-X — https://robotics-transformer-x.github.io/
- SimplerEnv — https://simpler-env.github.io/
- LIBERO — https://libero-project.github.io/
- Latent Diffusion — https://arxiv.org/abs/2112.10752
- TUM RGB-D — https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
