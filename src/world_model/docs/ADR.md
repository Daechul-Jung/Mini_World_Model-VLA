# Architecture Decision Records -- World Model Track

*Private. Gitignored.*

**Philosophy.** Every component that encodes an idea must be swappable without
touching anything else. Correctness before performance. One file, one
responsibility. Where a decision follows from the 7.7 GiB GPU rather than from
research judgement, say so.

---

### ADR-001: Architecture from Genie 1's paper, capability targets from Genie 3

**Decision.** Implement the published Genie 1 recipe (arXiv:2402.15391):
video tokenizer -> latent action model -> action-conditioned dynamics. Treat
Genie 3's blog-post capabilities as targets, not as a spec.

**Reason.** Genie 3 (August 2025) has no architecture paper. What is public is a
capability description: 720p, 24 fps, minutes of interaction with roughly a
minute of visual memory, promptable world events, no physics engine, and
autoregressive frame-by-frame generation. None of that is reproducible from the
blog post. Genie 1 has full architectural detail, ablations, and hyperparameters.

**Trade-off.** The reproduction is two generations behind on capability. That is
the honest position, and it is more useful than guessing at Genie 3's internals:
the gap between Genie 1's mechanisms and Genie 3's capabilities is where the
`memory/` and `physics/` slots live. Say "Genie-style, following Genie 1" in any
writeup, not "a Genie 3 reproduction".

---

### ADR-002: Four contracts, four stages, four checkpoints

**Decision.** `VideoTokenizer`, `LatentActionModel`, `Dynamics`, `Decoder` are
separate ABCs, each trained by its own `Stage` and saved to its own checkpoint
with recorded lineage.

**Reason.** Three things follow from it that end-to-end training does not give.
A component can be replaced without retraining the others' code. A component can
be *diagnosed* alone -- when rollouts are bad you can ask whether the tokenizer
reconstructs, whether the LAM discriminates, whether the dynamics beats copying,
and get three separate answers. And on 7.7 GiB, only one component is resident
at a time.

**Trade-off.** Stage-wise training is strictly weaker than joint training: the
tokenizer optimises reconstruction, not next-frame predictability, so it may
discard exactly what the dynamics model needs. Genie accepts the same trade for
the tokenizer. Mitigated for stages B/C by the co-training escape hatch (ADR-003).

---

### ADR-003: LAM as its own stage, with co-training as an opt-in

**Decision.** Stage B trains the latent action model standalone. Stage C's
`latent_action.freeze: false` continues training it alongside the dynamics model.

**Reason.** Genie co-trains the LAM and dynamics in phase 2. Co-training is
better -- the LAM learns to distinguish transitions the dynamics model actually
needs distinguished. But a standalone stage is where you discover cheaply that
your clips have no motion in them, or that the codebook has collapsed to two
codes, before spending a stage-C run finding out.

**Trade-off.** Two phases instead of one, and a standalone LAM optimum is not the
co-trained optimum. Run B standalone first, then C with `freeze: false` to
recover the paper's setup.

---

### ADR-004: The LAM takes pixels, not tokens

**Decision.** `VQLatentActionModel` owns its own patch embedding and reads raw
frames, rather than reusing the stage-A tokenizer's output.

**Reason.** Genie ablates exactly this (Table 2): pixel input scores 1.91
controllability vs 1.33 for token input. Tokenisation is trained for
reconstruction and discards fine motion cues, which are precisely the signal the
LAM is trying to isolate.

**Trade-off.** A second visual encoder, and stage B cannot reuse stage A's
compute. Both are small -- the LAM trunk is a few million parameters, because its
job is discovering which transitions exist, not rendering.

---

### ADR-005: Codebook health is a first-class stage-A metric

**Decision.** Stage A reports `codebook_use` and `perplexity` alongside PSNR, and
`best.pt` is selected on PSNR *with* usage watched. The quantizer is its own
registry slot.

**Reason.** The characteristic VQ-VAE failure is not bad reconstruction -- it is
good reconstruction achieved with 5% of the codebook, where the encoder routes
everything through a handful of codes and the decoder memorises. PSNR looks fine.
The dynamics model then has almost no vocabulary to predict and stage C is
wasted. In the first smoke run here, `codebook_use` was 0.30 after one epoch,
which is exactly the number that would have been invisible otherwise.

**Trade-off.** More metrics to watch. The alternative is discovering the problem
two stages later.

---

### ADR-006: Causal GPT as the dynamics baseline, MaskGIT as the known upgrade

**Decision.** Ship `causal_gpt` -- a flat causal sequence over all tokens of all
frames -- as the stage-C baseline, and document MaskGIT as the planned swap.

**Reason.** The flat model is simple and its training objective is correct. Its
two weaknesses are known and written into its module docstring: (1) sampling is
not coherent *within* a frame, because all h*w tokens are drawn from one forward
pass and are therefore conditionally independent; (2) cost is quadratic in
`T*h*w`. Genie fixes both -- MaskGIT decodes a frame in 25 iterative rounds each
conditioned on what is already committed, and the ST-transformer splits attention
into spatial and temporal.

**Trade-off.** The baseline will produce visibly incoherent frames. That is
acceptable for a baseline whose purpose is to be beaten, and it makes the
MaskGIT experiment a clean A/B. `research/004_maskgit_dynamics.md`.

---

### ADR-007: `delta_psnr` is the metric that decides whether stage C worked

**Decision.** Stage C reports Genie's Delta-PSNR -- PSNR of the prediction under
the true latent action minus PSNR under a random one -- as a validation metric.

**Reason.** Next-token cross-entropy on video is minimised well by copying the
previous frame, because most tokens do not change. A dynamics model that ignores
its action input entirely can have an excellent loss. `delta_psnr` near zero says
the action channel is dead, and nothing else in the training signal says it.
`copy_baseline_acc` is logged for the same reason.

**Trade-off.** Delta-PSNR needs the LAM at validation time, so the LAM cannot be
discarded until after stage C. Genie's scale is single digits (1.91 in the
paper), so treat > 0.5 as the first sign of life, not as a good score.

---

### ADR-008: 128 px, ~10 fps

**Decision.** Default resolution 128, `frame_skip: 3` on 30 fps TUM footage.

**Reason.** Resolution: at 256 px with three downsample stages a frame is
32x32 = 1024 tokens, so an 8-frame clip is 8192 sequence positions -- unaffordable
for the flat causal baseline on 7.7 GiB. 128 px gives 16x16 = 256 tokens/frame.
Frame rate: consecutive frames at 30 fps differ by almost nothing, and a LAM
trained on them correctly learns that the action is always "no-op". Genie trains
at 10 fps.

**Trade-off.** 128 px loses fine detail, and Genie 3's target is 720p. Frame rate
matters far more than resolution for a *world* model, so this is the right place
to spend the budget.

---

### ADR-009: Stage D trains on tokenizer latents, not dynamics predictions

**Decision.** The diffusion decoder is trained to render frozen stage-A latents.

**Reason.** Keeps one responsibility per stage. Training it on dynamics outputs
would make its loss depend on stage C's quality and let it learn to paper over
dynamics error -- which would then hide that error from every diagnostic.

**Trade-off.** At rollout time the decoder sees predicted latents, which are
slightly off-distribution from the tokenizer latents it trained on. Acceptable,
and the alternative couples two stages that need to stay separable.

---

### ADR-010: RL rollouts render with the tokenizer, not the diffusion decoder

**Decision.** `WorldModelEnv` defaults to `render="tokenizer"`.

**Reason.** The diffusion decoder costs ~25 network evaluations per frame. In an
RL loop with 32 parallel imagined episodes and 16 steps that is 12,800 extra
forwards per iteration, for image quality the policy does not obviously need.

**Trade-off.** The policy sees blurrier frames than a real camera would produce.
That gap is a real, unmeasured risk for a VLA pretrained on sharp real images,
and it is flagged as an open question rather than assumed away --
`research/012_reward_on_imagined_frames.md`.

---

### ADR-011: `memory/` and `physics/` are contracts before they are code

**Decision.** Both packages ship an ABC and a docstring laying out the design
space, with no implementation.

**Reason.** Both address problems that are not yet demonstrated in this repo.
Writing a spatial-memory module before a 64-step rollout has been observed to
fail means optimising against an imagined failure. The contract fixes the seam so
that when the failure *is* observed, the fix is a new file.

**Trade-off.** Empty packages look unfinished. They are the honest state:
"we know where this goes and we have not earned the right to build it yet."
