# PRD: VLA Track

*Private working document. Gitignored. Ask me to read it when you want me to reason about this project.*

---

## Goal

Build a vision-language-action research platform where **a new architectural idea
is a new file, not a new fork**. The eventual deliverable is a proposed VLA
architecture; everything here is scaffolding that makes proposing one cheap.

Two paths run in parallel, deliberately:

| Path | What it is | Why |
|------|-----------|-----|
| **From scratch** | Octo-style transformer, ~30-90M params, trained on an Open X-Embodiment pick-and-place subset | You understand every layer, can change anything, and can afford full training runs |
| **Pretrained + new layer** | OpenVLA-7B or pi0-3.3B loaded frozen, with a trainable module stack and head on top | Real manipulation competence is not reachable from 100 episodes; this borrows it |

The second path is where the research contribution lives. The first is how you
learn the machinery and get a controlled baseline.

---

## Owner

Daechul Jung. Single researcher, one machine, **RTX 4070 Laptop (7.7 GiB)**.
Every design decision in this project is downstream of that number.

---

## What exists now

| Component | Status |
|-----------|--------|
| `VLAPolicy` contract | done |
| `octo_torch` / `octo_small` / `octo_medium` | done, trains on real data |
| `openvla` frozen + adapter | written, not run against downloaded weights |
| `pi0` frozen + adapter | scaffold; `encode()` needs the chosen port's signature |
| Heads: `continuous_mse`, `gaussian`, `discrete_bins` | done |
| Modules: `bottleneck_adapter`, `gated_residual`, `wm_conditioning` | done |
| `openx_npz` dataset + action normalisation | done |
| `stage_bc` behaviour cloning | done, verified end-to-end |
| `stage_rl` RL post-training | contract only |
| Simulator evaluation | **not built -- the largest gap** |

The upstream JAX->PyTorch Octo port under `backbones/octo/components/` does not
import; `octo_torch` in `backbones/octo/policy.py` is the working implementation.
See `research/001_finish_octo_port.md`.

---

## Data

`data/openx/` -- 100 episodes x 50 steps, 256x256 RGB, **4-DoF actions**
(`dx, dy, dz, gripper`), three instruction phrasings. 459 MB.

Three consequences worth stating up front, because they shape what any result
here can mean:

1. **100 episodes is a pipeline dataset, not a training dataset.** Octo used
   800k trajectories across 25 datasets. Anything trained from scratch here will
   fit and then overfit. Use it to verify the machinery, not to claim
   performance.
2. **4-DoF is not 7-DoF.** OpenVLA and pi0 expect 7-DoF end-effector deltas.
   Mixing them requires an explicit adapter and is a known source of silent
   breakage, which is why `PolicySpec.action_dim` is checked against the dataset.
3. **The action normalisation statistics are part of the model.** They are
   computed once from the filtered episodes and saved into the checkpoint. A
   checkpoint without them emits numbers with no units.

---

## Success criteria

Ordered. Do not skip ahead -- each is a prerequisite for the next meaning anything.

- [x] **M0** `octo_small` trains on `data/openx/` and produces a checkpoint with
      its action normalisation attached.
- [ ] **M1** A simulator is wired up and reports a success rate.
      *Until this exists, every number in this project is an offline proxy.*
- [ ] **M2** `octo_small` beats a "repeat the previous action" baseline in sim.
- [ ] **M3** A pretrained backbone (pi0 preferred on this GPU) loads, runs frozen,
      and produces a zero-shot success rate.
- [ ] **M4** A `PolicyModule` inserted on top of a frozen backbone measurably beats
      the frozen-backbone-plus-head baseline. **This is the first real result.**
- [ ] **M5** RL post-training in a simulator beats the BC policy.
- [ ] **M6** RL post-training inside the world model beats BC, *measured in the
      simulator*. This is the project's headline claim.

---

## Out of scope, for now

- Real hardware. Everything is sim and offline data.
- Full or LoRA fine-tuning of a 7B model. Does not fit; see ADR-004.
- Multi-embodiment training. One action space at a time until M4.
- Training on the full Open X-Embodiment. Storage-bound.

---

## Constraints

- **GPU**: RTX 4070 Laptop, 7.7 GiB. bf16 everywhere; gradient accumulation
  instead of large batches; peak VRAM is logged every epoch.
- **Framework**: PyTorch only.
- **Storage**: a subset of one OpenX dataset. Any new dataset needs a stated size
  budget before download.
- **Modularity**: adding an idea must touch `modules/`, `heads/`, or `backbones/`
  plus one config -- never a training loop. If it does, the contract is wrong;
  fix the contract.
- **No commits without approval.**

---

## Open questions

Live, not rhetorical. Each has a research note.

1. **Which pretrained backbone?** OpenVLA at 4-bit is ~6-7 GB on a 7.7 GiB card.
   pi0 at 4-bit is ~2.5 GB. pi0 is probably the right call, but its PyTorch ports
   are less settled. -- `research/002_pi0_vs_openvla_frozen.md`
2. **Which simulator?** Real-robot-trained VLAs do not transfer to arbitrary
   MuJoCo scenes. SimplerEnv and LIBERO are the two that make sense.
   -- `research/003_simulation_stack.md`
3. **Does frozen-backbone + adapter actually beat a small from-scratch model on
   100 episodes?** Not obvious. Worth measuring before building on it.
   -- `research/004_frozen_vs_scratch.md`
4. **Which layer of a frozen backbone gives the best features?** The last layer is
   specialised for action-token prediction; an intermediate layer may transfer
   better. One config line to ablate.
