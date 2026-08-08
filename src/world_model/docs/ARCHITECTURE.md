# Architecture -- World Model Track

*Private. Gitignored.*

---

## Directory layout

```
src/world_model/
├── docs/                          # ADR / PRD / ARCHITECTURE / IMPROVEMENTS  (gitignored)
├── research/                      # idea specs -- write one BEFORE the code    (gitignored)
│
├── core/
│   ├── base.py                    # 4 ABCs + LatentSpec + ActionSpaceSpec  <-- THE contracts
│   ├── registry.py                # TOKENIZERS / QUANTIZERS / LATENT_ACTIONS / DYNAMICS / ...
│   └── genie.py                   # GenieWorldModel: composition root + imagine()
│
├── tokenizer/                     # STAGE A
│   ├── conv_vqvae.py              # per-frame conv VQ-VAE (baseline)
│   └── quantizers/
│       └── vq.py                  # straight-through VQ    <- swap slot for FSQ/LFQ
│
├── latent_action/                 # STAGE B
│   └── vq_lam.py                  # Genie LAM: pixel input, |A|=8, ST blocks
│
├── dynamics/                      # STAGE C
│   └── causal_gpt.py              # flat causal transformer (baseline)
│
├── decoder/                       # STAGE D  (optional)
│   ├── diffusion_decoder.py       # contract wrapper
│   ├── diffusion_unet.py          # the UNet
│   └── ddpm.py                    # noise schedule + DDIM step
│
├── memory/                        # <-- IDEA SLOT: long-horizon consistency (contract only)
│   └── base.py
├── physics/                       # <-- IDEA SLOT: geometry / physics priors  (contract only)
│   └── base.py
├── splatting/                     # 3DGS, kept as an explicit-geometry probe
│
├── data/
│   ├── image_dataset.py           # stills (stage A only)
│   └── video_dataset.py           # VideoFolderDataset (TUM), EpisodeNPZDataset (OpenX)
│
├── training/
│   ├── data.py
│   ├── stage_a_tokenizer.py
│   ├── stage_b_lam.py
│   ├── stage_c_dynamics.py
│   └── stage_d_decoder.py
│
└── eval/
    ├── recon.py                   # PSNR / SSIM / codebook usage / perplexity
    ├── controllability.py         # delta_psnr, action_sweep
    └── rollout.py                 # per-step decay, revisit_consistency, video export
```

---

## The four contracts

```python
class VideoTokenizer(nn.Module, ABC):        # STAGE A
    latent_spec: LatentSpec                  # grid, dim, discrete, vocab_size
    def encode(frames) -> {latents, indices, aux_loss}
    def decode(latents) -> frames
    def indices_to_latents(indices) -> latents
    def forward(frames) -> (recon, loss, metrics)

class LatentActionModel(nn.Module, ABC):     # STAGE B
    action_spec: ActionSpaceSpec             # kind="latent", num_actions
    def infer_actions(frames) -> {indices, embeddings, aux_loss}
    def forward(frames) -> (pred_next, loss, metrics)

class Dynamics(nn.Module, ABC):              # STAGE C
    latent_spec, action_spec
    accepts_discrete: bool
    def forward(tokens, actions) -> {loss, logits, token_acc}
    def predict_next(tokens, action, temperature) -> next_tokens
    def reset_cache()

class Decoder(nn.Module, ABC):               # STAGE D
    def render(latents, steps) -> frames
    def forward(frames, latents) -> (loss, metrics)
```

`GenieWorldModel` composes them and **asserts they agree at construction time** --
token grid, vocabulary size, discrete-vs-continuous, action-vocabulary size. A
mismatch raises immediately rather than three hours into stage C.

---

## Data flow -- training

```
STAGE A                     frames -> [tokenizer] -> recon
                                            |
                                       checkpoint A

STAGE B      frames (pixels, non-causal in time)
                    |
             [LAM encoder] -> VQ(|A|=8) -> a_t -> [LAM decoder] -> x_{t+1}
                    |
               checkpoint B                 (the decoder here is training-only)

STAGE C      frames --frozen A--> tokens (B,T,h,w)
             frames --frozen B--> a_t     (B,T-1)      or  dataset robot actions
                                     |
                          [dynamics] cross-entropy on next-frame tokens
                                     |
                               checkpoint C

STAGE D      frames --frozen A--> latents
                                     |
                          [diffusion decoder] denoising MSE
                                     |
                               checkpoint D
```

## Data flow -- rollout

```
real context frames -> [tokenizer.encode] -> token history
                                                  |
   action (latent code, or robot action) ---> [dynamics.predict_next]
                                                  |
                                            next tokens
                                                  |
                             ┌────────────────────┴───────────────────┐
                             |                                        |
                    tokenizer.decode                       decoder.render (25 steps)
                     fast, blurrier                          slow, sharp
                             |                                        |
                             └────────────────────┬───────────────────┘
                                                  |
                                        imagined frame -> RL env / video
```

---

## Stage dependency graph

```
   A (tokenizer)            B (latent action)
        |    \                    |
        |     \                   |
        |      \                  |
        v       v                 v
   D (decoder)   C (dynamics) <---┘
                       |
                       v
              GenieWorldModel.imagine()
                       |
                       v
          bridge/envs/world_model_env.py
```

B does **not** depend on A -- it reads pixels (ADR-004). C depends on both. D
depends only on A and is optional.

---

## Diagnostics, and what each failure means

The reason for stage-wise training is that these questions have separate answers.

| Symptom | Metric | Likely cause | Fix |
|---------|--------|--------------|-----|
| Blurry reconstructions | `val/psnr` low | tokenizer under-trained or too small | more epochs, more channels |
| Good PSNR, useless tokens | `codebook_use` < 0.2 | codebook collapse | different quantizer (FSQ/LFQ) |
| LAM learns nothing | `actions_used` <= 2 | no motion between frames | raise `data.frame_skip` |
| Dynamics "works", rollouts static | `token_acc` ≈ `copy_baseline_acc` | model learned to copy | longer clips, harder masking |
| Rollouts ignore actions | `delta_psnr` ≈ 0 | action channel dead | check stage B first, then conditioning |
| Frames incoherent within a frame | visual | independent token sampling | MaskGIT (ADR-006) |
| Long rollouts drift to mush | `psnr_decay` large | compounding one-step error | scheduled sampling, diffusion forcing |
| Long rollouts sharp but wrong room | `revisit_psnr` low | finite context | `memory/` slot |

---

## Adding an idea

Same pattern as the VLA track.

1. Write `research/NNN_your_idea.md` -- hypothesis, measurement, kill criterion.
2. Implement against the relevant ABC, in one file:
   ```python
   @DYNAMICS.register("maskgit_st", paper="arXiv:2402.15391", status="idea")
   class MaskGITSTDynamics(Dynamics):
       accepts_discrete = True
       ...
   ```
3. Point a config at it -- nothing else changes:
   ```bash
   python scripts/train/train_world_model.py --stage c --config genie_small.yaml \
       --set dynamics.name=maskgit_st --tokenizer_ckpt stage_a_tokenizer:best \
       --latent_action_ckpt stage_b_latent_action:best
   ```
4. `pytest tests/test_world_model.py` -- checks contract conformance for every
   registered component.
5. Record the result in `docs/IMPROVEMENTS.md`, including negatives.

---

## Checkpoints

```
checkpoints/world_model/
    stage_a_tokenizer/<run>/{config.yaml, manifest.json, best.pt, last.pt, metrics.jsonl}
    stage_b_latent_action/<run>/...
    stage_c_dynamics/<run>/...
    stage_d_decoder/<run>/...
```

Reference an earlier stage as `stage_a_tokenizer:best` -- resolved to the newest
run's `best.pt`. Every `.pt` records `frozen_parents`, so
`resolve_lineage(path)` answers "which tokenizer produced the tokens this
dynamics model was trained on" -- the question that silently ruins staged
training when nobody can answer it.

---

## Component registry

`python scripts/tools/list_components.py`:

| Registry | Names |
|----------|-------|
| `TOKENIZERS` | `conv_vqvae` |
| `QUANTIZERS` | `vq` |
| `LATENT_ACTIONS` | `vq_lam` |
| `DYNAMICS` | `causal_gpt` |
| `DECODERS` | `diffusion_unet` |
| `WM_DATASETS` | `images`, `video_folder`, `episode_npz` |
| `STAGES` | `stage_a_tokenizer`, `stage_b_latent_action`, `stage_c_dynamics`, `stage_d_decoder` |
