# Claude working notes for this repo

*Gitignored. This file orients Claude at the start of a session.*

## What this project is

Two research tracks plus a bridge between them, on one RTX 4070 Laptop
(**7.7 GiB VRAM** -- not 12 GB; size everything against this).

| Track | Goal |
|-------|------|
| `src/vla/` | Vision-language-action policies. From-scratch Octo-style, plus frozen pretrained backbones (OpenVLA / pi0) with new layers on top. |
| `src/world_model/` | Genie-style generative world model of rooms, trained component by component. |
| `src/bridge/` | RL post-training of the VLA inside world-model-generated environments. |

The long-term goal is to propose a new VLA architecture and a new world-model
architecture, and to combine them.

## Read these before answering architecture questions

They are gitignored and they hold the actual reasoning. **Read them when asked
about design, direction, or new ideas** -- do not answer from the code alone.

```
src/vla/docs/{PRD,ADR,ARCHITECTURE,IMPROVEMENTS}.md
src/world_model/docs/{PRD,ADR,ARCHITECTURE,IMPROVEMENTS}.md
src/bridge/docs/{PRD,ADR,ARCHITECTURE,IMPROVEMENTS}.md
src/{vla,world_model,bridge}/research/*.md
```

- **PRD** -- goal, milestones, open questions
- **ADR** -- why each decision was made, and its trade-off
- **ARCHITECTURE** -- layout, contracts, data flow, how to add an idea
- **IMPROVEMENTS** -- measured results, including negative ones
- **research/** -- one file per idea, each with a **kill criterion**

## House rules

1. **Everything is swappable through a registry.** Adding an idea means one new
   file plus one config line. If a change would touch a training loop, the
   contract is wrong -- fix the contract, do not special-case the loop.
2. **Write the research note before the code.** Especially the kill criterion.
3. **Train component by component.** Each stage loads its predecessors frozen and
   writes one checkpoint with recorded lineage. No end-to-end training.
4. **Record negative results** in IMPROVEMENTS.md. They are worth as much as
   positive ones and they stop ideas being retried.
5. **Do not commit without asking.**
6. **Do not publish the docs.** They are gitignored on purpose.

## Things that are true and easy to forget

- **Genie 3 has no architecture paper.** The implementation follows Genie 1
  (arXiv:2402.15391); Genie 3's blog-post capabilities are targets. Say
  "Genie-style, following Genie 1", never "a Genie 3 reproduction".
- **LSUN cannot train stages B or C.** Unordered stills of different rooms have
  no next frame. Stage A only.
- **A latent action model's 8 discrete codes cannot express a 7-DoF robot
  action.** For robotics, train stage C with `action_kind=robot` on OpenX rather
  than translating.
- **Offline action error does not predict task success.** No simulator is wired
  up yet; this is the biggest gap in the project (`vla/research/003`).
- **OpenVLA cannot be fine-tuned on this GPU.** Frozen 4-bit + adapter is the
  only mode that fits, and it is tight. pi0 is the more comfortable option.
- **The upstream Octo JAX port under `backbones/octo/components/` does not
  import.** `backbones/octo/policy.py` is the working implementation.

## Commands

```bash
python scripts/tools/list_components.py          # everything registered, by name
python -m pytest tests/ -q                       # contract tests, CPU, seconds

python scripts/train/train_world_model.py --stage a --config genie_small_lsun.yaml
python scripts/train/train_world_model.py --stage b --config genie_small.yaml
python scripts/train/train_world_model.py --stage c --config genie_small.yaml \
    --tokenizer_ckpt stage_a_tokenizer:best --latent_action_ckpt stage_b_latent_action:best

python scripts/train/train_vla.py --stage bc --config octo_small.yaml
```

`--set key.path=value` overrides any config value inline and parses as YAML.
