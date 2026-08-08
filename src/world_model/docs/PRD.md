# PRD: World Model Track

*Private working document. Gitignored. Ask me to read it when you want me to reason about this project.*

---

## Goal

Reproduce a small Genie-style generative world model, component by component,
with every component replaceable -- so that a new idea (JEPA-style latent
prediction, explicit memory, physics priors) is a new file rather than a fork.

Two deliverables, and they are different:

1. **A generative model of rooms.** Prompt with a few real frames, roll forward
   under actions, get a coherent short video. Verifies the pipeline.
2. **An environment.** The same model driven by a VLA's actions, producing
   observations and rewards for RL post-training. This is the research goal, and
   it imposes requirements the first deliverable does not -- most importantly a
   usable action interface (see Open Question 1).

---

## Which Genie is this based on?

**Genie 3 has no published architecture.** DeepMind's August 2025 release is a
blog post describing capabilities -- 720p, 24 fps, a few minutes of interaction,
visual memory reaching roughly one minute back, "promptable world events", no
physics engine, autoregressive frame-by-frame generation -- with no paper
describing how. There is nothing to reproduce from it directly.

**Genie 1 (arXiv:2402.15391) has a full paper**, and that is what this project
implements:

| Component | Genie 1 | Here |
|-----------|---------|------|
| Video tokenizer | ST-transformer VQ-VAE, 1024 codes, patch 4 | conv VQ-VAE (baseline); ST version is a planned swap |
| Latent action model | pixel input, patch 16, \|A\| = 8 VQ codes | same |
| Dynamics | MaskGIT, mask rate U(0.5, 1), 25 steps, temp 2.0 | causal GPT (baseline); MaskGIT is a planned swap |
| Resolution / fps | 160x90, 10 fps | 128x128, ~10 fps via `frame_skip` |
| Scale | 11B, 30k hours | ~10-50M, hours |

So: **architecture from Genie 1, capability targets from Genie 3.** The gap
between them -- long-horizon consistency, real-time speed, physical plausibility
-- is not a bug list, it is the research programme. `memory/` and `physics/` exist
because closing it needs mechanisms Genie 1 does not have.

One paper detail worth restating because it is easy to get wrong: Genie trains
the tokenizer **first**, then **co-trains** the LAM and the dynamics model. This
project trains the LAM as its own stage so it gets an inspectable checkpoint, and
exposes `latent_action.freeze: false` in stage C to recover the paper's
co-training as a second phase. See ADR-003.

---

## Owner

Daechul Jung. RTX 4070 Laptop, **7.7 GiB**. Every size decision follows from it.

---

## What exists now

| Stage | Component | Status |
|-------|-----------|--------|
| A | `conv_vqvae` tokenizer + `vq` quantizer | done, trains on LSUN + TUM |
| B | `vq_lam` latent action model | done, trains on TUM video |
| C | `causal_gpt` dynamics | done, trains on frozen tokens |
| D | `diffusion_unet` decoder | done |
| -- | `GenieWorldModel.imagine()` | done |
| -- | Controllability / rollout / revisit metrics | done |
| -- | `memory/`, `physics/` | contracts only |
| -- | ST-transformer tokenizer, MaskGIT dynamics | not built |

---

## Data, and what each source can and cannot train

This is the single most important table in this document.

| Source | Size | Stage A | Stage B | Stage C | Notes |
|--------|------|:-------:|:-------:|:-------:|-------|
| `data/lsun_rooms/` | 5000 stills, 99 MB | yes | **no** | **no** | Unordered photos of *different* rooms. There is no next frame. |
| `data/tum_rgbd/` | 2 sequences, 4.2 GB | yes | yes | yes | Real camera motion + depth + ground-truth poses. The main source. |
| `data/openx/` | 100 episodes, 459 MB | yes | yes | yes | The only source with **labelled actions**. |

**LSUN cannot train a latent action model or a dynamics model.** Running stage B
or C against it is a category error, not a tuning problem -- there is no temporal
structure to learn. Use it for stage A only, where its visual variety improves
codebook coverage.

TUM is ~30 fps. Consecutive frames barely differ, so a LAM trained on them learns
that every action is a no-op. `frame_skip: 3` brings it to ~10 fps, matching
Genie. This is a real setting, not a detail.

---

## Success criteria

- [x] **W0** All four stages train and chain checkpoints correctly.
- [ ] **W1** Stage A: `val/psnr > 22` **and** `val/codebook_use > 0.5`.
      Both, not either -- see ADR-005.
- [ ] **W2** Stage B: `actions_used >= 6` of 8 and `action_perplexity > 4`.
      Below that, the bottleneck has collapsed and stage C is pointless.
- [ ] **W3** Stage C: `token_acc` clearly above `copy_baseline_acc`.
- [ ] **W4** Stage C: `delta_psnr > 0.5`. **This is the first result that means
      anything** -- it says actions actually control the model.
- [ ] **W5** An 8-step rollout from real context frames is recognisably the same
      room. Qualitative, judged by eye, written up in IMPROVEMENTS.
- [ ] **W6** `action_sweep` shows the 8 codes doing visibly different things.
- [ ] **W7** A 64-step rollout does not collapse -- this is where `memory/` earns
      its keep.
- [ ] **W8** An action-conditioned (`action_kind=robot`) model trained on OpenX
      responds correctly to real robot actions. Prerequisite for the bridge.

---

## Out of scope, for now

- 720p or 24 fps. Genie 3's numbers, DeepMind's hardware.
- Training on internet-scale video.
- Text-prompted world generation ("promptable world events"). Needs a text
  encoder and paired data neither of which exists here.
- 3D Gaussian Splatting as the world representation. `splatting/` is kept as an
  explicit-geometry probe and as a possible `memory/` implementation, but the
  main line is Genie's implicit approach.

---

## Constraints

- **GPU**: 7.7 GiB. 128 px, not 256: at 256 px with 3 downsample stages a frame
  is 32x32 = 1024 tokens and an 8-frame clip is 8192 positions, which the causal
  GPT baseline cannot afford. 128 px gives 256 tokens/frame.
- **Storage**: TUM RGB-D is already 4.2 GB. New video sources need a budget first.
- **Stage isolation**: each stage loads its predecessors frozen and writes one
  checkpoint. No end-to-end training.
- **No commits without approval.**

---

## Open questions

1. **How does a VLA drive this?** A Genie world model takes 8 discrete latent
   codes; a VLA emits continuous 4-7 DoF actions. These are unrelated spaces.
   The likely answer is to sidestep it -- train stage C with `action_kind=robot`
   on OpenX so the dynamics model is conditioned on real actions from the start.
   -- `src/bridge/docs/ADR.md` and `research/011_unified_action_conditioning.md`
2. **Where does long-horizon consistency come from?** Not from a longer context:
   one minute at 10 fps and 256 tokens/frame is 150k positions. Compression,
   retrieval, or explicit spatial state. -- `research/007_long_horizon_memory.md`
3. **Can this data teach physics at all?** Genie 3 learns physics from
   internet-scale video. With 5k stills and two desk sequences, probably not
   without injected structure -- depth and pose auxiliary heads, using the
   supervision TUM already ships. -- `research/008_physics_auxiliary_heads.md`
4. **Is discrete tokenisation the right choice?** It is Genie's, and it makes
   MaskGIT possible. A continuous latent + JEPA-style predictor avoids codebook
   collapse entirely and may suit the small-data regime better.
   -- `research/009_jepa_dynamics.md`
