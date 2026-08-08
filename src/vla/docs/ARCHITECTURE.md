# Architecture -- VLA Track

*Private. Gitignored.*

---

## Directory layout

```
src/vla/
├── docs/                          # ADR / PRD / ARCHITECTURE / IMPROVEMENTS  (gitignored)
├── research/                      # idea specs -- write one BEFORE writing code (gitignored)
│
├── core/
│   ├── base.py                    # VLAPolicy ABC + PolicySpec   <-- THE contract
│   └── registry.py                # POLICIES / HEADS / MODULES / VLA_DATASETS / RL_ALGORITHMS
│
├── backbones/                     # the big trunk -- swappable
│   ├── octo/
│   │   ├── policy.py              # WORKING: octo_torch / octo_small / octo_medium
│   │   ├── octo_module.py         # partial JAX port -- does not import, not used
│   │   ├── components/            #   "
│   │   └── UPSTREAM.md            # provenance
│   ├── openvla/
│   │   ├── loader.py              # HF load, 4-bit, LoRA (documented as out of reach)
│   │   └── policy.py              # frozen backbone + modules + head
│   └── pi0/
│       ├── loader.py
│       └── policy.py              # scaffold: encode() needs the chosen port's signature
│
├── modules/                       # <-- THE IDEA SLOT
│   ├── base.py                    # PolicyModule ABC, ModuleStack, build_modules
│   ├── adapter.py                 # bottleneck_adapter, gated_residual
│   └── wm_conditioning.py         # cross-attend to world-model imagination
│
├── heads/                         # features -> actions, and the matching loss
│   ├── base.py                    # ActionHead ABC
│   ├── continuous.py              # continuous_mse, gaussian (RL-capable)
│   └── discrete.py                # discrete_bins (RT-1 / OpenVLA style)
│
├── data/
│   └── openx_dataset.py           # OpenXWindowDataset + ActionSpec computation
│
├── training/
│   ├── data.py                    # loaders; splits BY EPISODE
│   └── stage_bc.py                # behaviour cloning (scratch or frozen+adapter)
│
├── rl/
│   └── base.py                    # RLAlgorithm ABC, Rollout
│
└── eval/
    └── offline.py                 # action error, gripper accuracy, transition accuracy
```

---

## The contract

```python
class VLAPolicy(nn.Module, ABC):
    @property
    def spec(self) -> PolicySpec: ...

    def encode(self, obs: Observation) -> Tensor:            # (B, T, D)
    def forward(self, obs: Observation) -> Action:
    def loss(self, obs, target_actions) -> tuple[Tensor, dict]:

    # provided
    def act(self, obs, action_spec) -> Tensor:               # physical units
    def trainable_parameters(self) -> list[Parameter]:
    def freeze_backbone(self, freeze=True) -> None:
```

`encode` is separate from `forward` on purpose. It is the tensor `modules/` hook
into, the tensor an RL critic reads, and the tensor a world model can condition
on. A frozen backbone runs it under `no_grad`; a trainable one does not.

---

## Data flow

```
        Observation(image (B,T,3,H,W), instruction: list[str])
                              |
                    ┌─────────▼──────────┐
                    │     BACKBONE       │   frozen (OpenVLA/pi0) or trained (Octo)
                    │  encode() -> (B,T,D)│
                    └─────────┬──────────┘
                              |
                    ┌─────────▼──────────┐
                    │   MODULE STACK     │   <-- your idea goes here
                    │  identity at init  │       shape-preserving, stackable
                    │  context: {wm_latents, goal, ...}
                    └─────────┬──────────┘
                              |
                    ┌─────────▼──────────┐
                    │       HEAD         │   owns loss() and sample()
                    │ mse | gaussian | bins | diffusion
                    └─────────┬──────────┘
                              |
              Action(continuous (B, chunk, A), logp, latent)
                              |
                    ActionSpec.denormalize  -> metres, radians, gripper
                              |
                    Simulator  or  WorldModelEnv
```

---

## Adding an idea

The whole point of the layout. Five steps, none of which touch a training loop.

1. **Write the spec first**: `research/NNN_your_idea.md` -- hypothesis, what you
   will measure, and the kill criterion. Writing the kill criterion before the
   code is what stops a dead idea consuming three weeks.
2. **Implement** `modules/your_idea.py`:
   ```python
   @MODULES.register("your_idea", status="idea")
   class YourModule(PolicyModule):
       required_context = ("something_you_need",)

       def __init__(self, dim, ...):
           super().__init__(dim)
           self.out = nn.Linear(dim, dim)
           nn.init.zeros_(self.out.weight)   # identity at init -- required
           nn.init.zeros_(self.out.bias)

       def forward(self, features, context=None):
           ctx = self.check_context(context)
           return features + self.out(...)
   ```
3. **Run it**: no code changes, just config.
   ```bash
   python scripts/train/train_vla.py --stage bc --config octo_small.yaml \
       --set 'policy.modules=[{"name":"your_idea"}]'
   ```
4. **Verify the contract**: `pytest tests/test_vla_modules.py` -- checks identity
   at init and shape preservation for every registered module.
5. **Record the result** in `docs/IMPROVEMENTS.md`, including when it did not
   work. A module whose gate stays at zero is a result.

---

## Checkpoints

```
checkpoints/vla/stage_bc/<run_name>/
    config.yaml       exact resolved config
    manifest.json     every checkpoint written, with metrics and lineage
    best.pt           best by val/action_l1
    last.pt
    metrics.jsonl     per-step scalars
```

Each `.pt` holds `{state: {model, optimizer, action_spec}, meta: {...}}`, where
`meta` records `component`, `stage`, `config_hash`, and `frozen_parents`. Load a
previous run with `--init_ckpt stage_bc:best`; `resolve_lineage()` walks the
parent chain.

---

## Component registry

`python scripts/tools/list_components.py` prints everything registered. Current
VLA entries:

| Registry | Names |
|----------|-------|
| `POLICIES` | `octo_torch`, `octo_small`, `octo_medium`, `openvla`, `pi0` |
| `HEADS` | `continuous_mse`, `gaussian`, `discrete_bins` |
| `MODULES` | `bottleneck_adapter`, `gated_residual`, `wm_conditioning` |
| `VLA_DATASETS` | `openx_npz` |
| `STAGES` | `stage_bc` |

---

## What this architecture deliberately does not solve

Stated so it is not rediscovered as a surprise:

- **Per-layer injection into a frozen backbone.** `PolicyModule` sits between the
  trunk and the head. LoRA-style adapters inside attention blocks need
  backbone-specific code.
- **Multi-embodiment action spaces.** One `PolicySpec.action_dim` per policy.
  Cross-embodiment training needs an action-space adapter that does not exist yet.
- **Streaming / real-time inference.** Everything is batched offline. `reset()`
  exists on the contract for per-episode state but nothing implements a KV cache.
