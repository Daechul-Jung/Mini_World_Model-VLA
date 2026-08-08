# Architecture Decision Records -- VLA Track

*Private. Gitignored.*

**Philosophy.** Every component that encodes an idea must be swappable without
touching anything else. Correctness before performance. One file, one
responsibility. When a decision is forced by the 7.7 GiB GPU rather than by
research judgement, say so explicitly -- otherwise it gets mistaken for a
principle later.

---

### ADR-001: Three-part decomposition -- backbone / modules / head

**Decision.** Every policy is `backbone -> modules -> head`, with each part
independently swappable through a registry.

**Reason.** The stated goal is to add new layers to an existing VLA. That
requires a named place for "the new layer" that is neither inside the pretrained
trunk nor inside the action decoder. `modules/` is that place. It also separates
the two things people conflate: changing *how actions are parameterised* (head)
and changing *what the policy computes* (modules).

**Trade-off.** Some ideas do not fit the seam -- anything that needs to modify
attention *inside* the backbone (adapters injected per-layer, LoRA) cannot be a
`PolicyModule`. Those need backbone-specific surgery. The contract is honest
about covering the between-trunk-and-head case only.

---

### ADR-002: Modules must be identity at initialisation

**Decision.** A newly constructed `PolicyModule` returns its input unchanged --
zero-initialised output projections, gates starting at zero. Enforced by a test.

**Reason.** This is what makes "frozen pretrained backbone + new idea" trainable
at all. At step 0 the policy behaves exactly like the pretrained one, so training
starts from the pretrained loss rather than from random. Without it, a randomly
initialised module destroys the backbone's representation in the first few
hundred steps and the pretrained weights have bought nothing.

**Trade-off.** Constrains module design: no module can be a plain `nn.Linear`
stack. The gate also has to be watched -- one that never leaves zero means the
idea is inert, which is useful information but has to be looked for.

---

### ADR-003: The head owns the loss, not the trainer

**Decision.** `ActionHead.loss()` computes the behaviour-cloning objective.
`stage_bc` calls `policy.loss()` and knows nothing about action parameterisation.

**Reason.** Regression, discrete bins, diffusion and flow matching need four
different objectives over the same data. Putting the loss in the trainer would
mean an `if head_type ==` chain that grows with every new head.

**Trade-off.** A head is now a bigger object than "a linear layer". Worth it: the
choice of head is a real research variable here, not an implementation detail --
MSE cannot represent multi-modal demonstrations, and that is the most likely
reason a policy with good offline error fails at grasp points.

---

### ADR-004: On this GPU, large backbones are frozen feature extractors

**Decision.** OpenVLA and pi0 are loaded 4-bit and frozen. Only `modules/` and
the head train. LoRA is implemented but documented as out of reach.

**Reason.** Arithmetic, not preference. OpenVLA-7B: full fine-tuning ~150 GB
(8xA100 in the paper), LoRA r=32 ~27 GB (1xA100 reported), 4-bit weights ~5-6 GB.
pi0-3.3B at 4-bit is ~2.5 GB. The card has 7.7 GiB. Frozen + adapter is the only
mode that fits, and for OpenVLA even that is tight.

**Trade-off.** Frozen features cap what can be learned -- if the pretrained
representation does not contain the information a task needs, no adapter recovers
it. This must be stated as a limitation in any writeup, and it is *not* the same
claim as "we fine-tuned OpenVLA". It also makes pi0 the more practical of the two
backbones here, despite OpenVLA being the more standard baseline.

---

### ADR-005: Rewrite Octo in PyTorch rather than finish the JAX port

**Decision.** `backbones/octo/policy.py` implements the Octo *recipe* directly.
The partial line-by-line port of the upstream Flax code is kept but not imported.

**Reason.** The port carries Flax's dataclass-attribute idiom (`kernel_init:
Callable = nn.init.xavier_uniform()`), which executes at class-definition time in
Python and raises. Fixing it is a real project, and it only pays off if the goal
is loading official `rail-berkeley/octo-*` checkpoints -- which additionally
requires exact parameter-name correspondence.

**Trade-off.** No access to pretrained Octo weights. Given that the pretrained
path is being served by OpenVLA/pi0 anyway, and that a from-scratch Octo is
meant as a *controlled* baseline you fully understand, this is the cheaper
allocation. Revisit if pretrained Octo specifically is needed --
`research/001_finish_octo_port.md`.

---

### ADR-006: Action normalisation statistics travel with the checkpoint

**Decision.** `ActionSpec` (per-dimension q01/q99, mean/std, gripper index) is
computed from the training episodes and written into every checkpoint via
`Stage.extra_state()`.

**Reason.** A policy emits values in [-1, 1]. Turning those into metres and
radians requires the exact statistics of the exact episodes it trained on. A
checkpoint without them is not deployable, and -- worse -- silently produces
plausible-looking wrong actions when paired with the wrong statistics. This is
the single most common cause of "the model trained fine but does nothing on the
robot".

**Trade-off.** Changing the dataset filter changes the statistics, so checkpoints
from different filters are not interchangeable. That is correct behaviour, and
`config_hash` in the checkpoint makes the mismatch detectable.

---

### ADR-007: Gripper accuracy is tracked separately from action error

**Decision.** Every head and every evaluation reports gripper accuracy, and
`continuous_mse` supports a `gripper_weight` above 1.

**Reason.** The gripper is one dimension of four (or seven), binary, and decides
task success. Averaged into a mean L1 it is invisible: a policy can have
excellent aggregate error and open the gripper at the wrong moment on every
episode. `gripper_transition_acc` -- accuracy restricted to the timesteps where
the gripper state *changes* -- is stricter still, and is the number closest to
predicting success.

**Trade-off.** Up-weighting the gripper trades continuous-dimension accuracy for
it. That is usually the right trade and should still be chosen deliberately.

---

### ADR-008: RL post-training requires a stochastic head, and this is enforced

**Decision.** `RLAlgorithm.__init__` raises if `head.supports_rl` is False.
`continuous_mse` is deliberately not RL-capable.

**Reason.** A deterministic regression head has no action density, so no policy
gradient exists. Without the check, the most likely outcome is a run that
executes, produces loss curves, and optimises nothing -- the most expensive kind
of bug. Use `gaussian` (Gaussian NLL for BC, tanh-corrected log-prob for RL), so
a single checkpoint can be BC-pretrained and RL-finetuned without swapping heads.

**Trade-off.** BC with a Gaussian head is slightly worse-conditioned than plain
regression. Worth it to keep the pretrain-then-post-train path unbroken.

---

### ADR-009: Simulator choice is SimplerEnv / LIBERO, not plain MuJoCo

**Decision.** Evaluation targets SimplerEnv and LIBERO. Plain
Gymnasium/MuJoCo tasks are not a target.

**Reason.** OpenVLA, Octo and pi0 are trained on real-robot data (Open
X-Embodiment, Bridge, RT-1). They do not transfer zero-shot to arbitrary MuJoCo
scenes -- the visual and action distributions are unrelated. SimplerEnv exists
specifically to evaluate real-robot-trained VLAs in simulation, with visual
matching to the Bridge and Google-Robot setups. LIBERO has official OpenVLA
fine-tuned checkpoints, which gives a reproducible reference number.

**Trade-off.** Both are heavier dependencies than Gymnasium, and SimplerEnv needs
SAPIEN. The alternative -- a MuJoCo number that means nothing -- is worse.
`research/003_simulation_stack.md`.

---

### ADR-010: Split train/val by episode, never by window

**Decision.** `build_vla_loaders` holds out whole episodes.

**Reason.** Consecutive frames within an episode are nearly identical. A
window-level split puts near-duplicates on both sides, and validation error then
measures memorisation while reporting generalisation. With 100 episodes this
error would be large enough to invalidate every comparison made on top of it.

**Trade-off.** With few episodes, a 10% episode split is a small and noisy
validation set. Noisy-but-honest beats precise-but-wrong.
