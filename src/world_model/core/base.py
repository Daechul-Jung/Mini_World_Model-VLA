"""The four world-model component contracts.

The architecture follows the *published* Genie recipe (Bruce et al., 2024,
arXiv:2402.15391), because Genie 3 has no public architecture paper -- only a
capability blog post. See `src/world_model/docs/ADR.md`, ADR-001.

    frames ──► VideoTokenizer ──► discrete/continuous latents ─┐
                                                               ├─► Dynamics ──► next latents ──► Decoder ──► pixels
    frames ──► LatentActionModel ──► latent action a_t ────────┘

Four contracts, four stages, four checkpoints:

| Stage | Component          | Trained on           | Frozen inputs        |
|-------|--------------------|----------------------|----------------------|
| A     | `VideoTokenizer`   | single frames/clips  | --                   |
| B     | `LatentActionModel`| frame pairs (pixels) | --                   |
| C     | `Dynamics`         | token clips          | tokenizer, (LAM)     |
| D     | `Decoder`          | tokens -> pixels     | tokenizer            |

Rules that make components swappable:

1. A component never imports a *sibling* implementation -- only these ABCs.
2. Shapes are fixed by the ABC docstrings, not by whichever impl was written first.
3. `LatentSpec` / `ActionSpaceSpec` are how a component advertises its interface
   so `GenieWorldModel` can assert compatibility at construction time instead of
   crashing three hours into stage C.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------- specs


@dataclass(frozen=True)
class LatentSpec:
    """What one frame becomes after tokenisation.

    `discrete=True`  -> tokens are integer indices in [0, vocab_size).
    `discrete=False` -> tokens are continuous vectors of width `dim` (the JEPA /
                        continuous-diffusion path). A dynamics model declares
                        which it supports via `accepts_discrete`.
    """

    grid: Tuple[int, int]          # (h, w) tokens per frame
    dim: int                       # embedding width per token
    discrete: bool = True
    vocab_size: Optional[int] = None

    @property
    def tokens_per_frame(self) -> int:
        return self.grid[0] * self.grid[1]

    def assert_compatible(self, other: "LatentSpec", what: str = "component") -> None:
        if (self.grid, self.discrete) != (other.grid, other.discrete):
            raise ValueError(f"{what} latent spec mismatch: {self} vs {other}")
        if self.discrete and self.vocab_size != other.vocab_size:
            raise ValueError(f"{what} vocab mismatch: {self.vocab_size} vs {other.vocab_size}")


@dataclass(frozen=True)
class ActionSpaceSpec:
    """The action interface the dynamics model is conditioned on.

    Three kinds coexist in this project and confusing them is the single most
    likely source of silent breakage:

    * `latent`   -- Genie's unsupervised codes, an integer in [0, num_actions).
                    Produced by the LAM at training time, chosen by a user or an
                    adapter at inference. The LAM is discarded at inference.
    * `robot`    -- a real continuous robot action (e.g. 4-DoF UCSD delta,
                    7-DoF Bridge). This is what the VLA emits.
    * `none`     -- unconditional video prediction, for stage-C smoke tests.

    Coupling a VLA to the world model means bridging `robot` -> `latent`; that
    translation lives in `src/bridge/action_space.py`, never inside a dynamics
    model.
    """

    kind: str = "latent"           # "latent" | "robot" | "none"
    num_actions: Optional[int] = None   # for kind="latent"
    dim: Optional[int] = None           # for kind="robot"

    def __post_init__(self) -> None:
        if self.kind not in ("latent", "robot", "none"):
            raise ValueError(f"unknown action kind {self.kind!r}")


# ---------------------------------------------------------------- components


class VideoTokenizer(nn.Module, ABC):
    """Stage A: frames <-> latents.

    Genie uses a VQ-VAE with ST-transformer blocks so tokens see temporal
    context. A per-frame convolutional VQ-VAE is a valid (weaker) drop-in; both
    satisfy this contract, which is why stage A is swappable in one config line.
    """

    @property
    @abstractmethod
    def latent_spec(self) -> LatentSpec: ...

    @abstractmethod
    def encode(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames: (B, T, 3, H, W) in [-1, 1].

        Returns a dict with at least:
          `latents` (B, T, D, h, w) -- continuous (post-quantisation) latents
          `indices` (B, T, h, w)    -- integer codes, discrete tokenizers only
          `aux_loss` scalar         -- commitment/entropy loss, 0 if none
        """

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (B, T, D, h, w) -> frames (B, T, 3, H, W) in [-1, 1]."""

    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, T, h, w) -> latents (B, T, D, h, w) via the codebook.

        Required by any discrete tokenizer, because the dynamics model emits
        indices while the stage-D decoder conditions on latents.
        """
        raise NotImplementedError(f"{type(self).__name__} is not a discrete tokenizer")

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, T, h, w) -> frames. Default: codebook lookup then decode."""
        return self.decode(self.indices_to_latents(indices))

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Reconstruction pass for stage A: returns (recon, loss, metrics)."""


class LatentActionModel(nn.Module, ABC):
    """Stage B: infer *what changed* between frames, without action labels.

    Genie's LAM encodes (x_{<=t}, x_{t+1}) -> a_t and decodes (x_{<=t}, a_t) ->
    x_{t+1}, with a VQ bottleneck of |A| = 8 codes forcing a_t to carry only the
    controllable part of the transition. It takes **raw pixels**, not tokens --
    the paper's ablation shows tokenisation destroys motion information the LAM
    needs (Table 2: pixel-input beats token-input on controllability).

    The LAM is discarded at inference. Anything that later wants to *drive* the
    world model supplies a code index directly.
    """

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpaceSpec: ...

    @abstractmethod
    def infer_actions(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames: (B, T, 3, H, W) -> latent actions for the T-1 transitions.

        Returns at least:
          `indices`   (B, T-1)      -- discrete latent action per transition
          `embeddings`(B, T-1, D)   -- their embeddings, for conditioning
          `aux_loss`  scalar
        """

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Stage-B pass: predict x_{t+1} through the bottleneck.

        Returns (predicted_next_frames, loss, metrics). The decoder used here is
        a *training-only* head; it is not the stage-D decoder.
        """


class Dynamics(nn.Module, ABC):
    """Stage C: predict the next frame's latents given history and an action.

    Genie uses MaskGIT: bidirectional attention *within* a frame with parallel
    iterative decoding, causal attention *across* frames. A plain causal GPT over
    flattened tokens also satisfies this contract but decodes one token at a
    time, which is ~1000x more forward passes per frame at a 32x32 grid.
    """

    #: whether this model consumes integer tokens (True) or continuous latents
    accepts_discrete: bool = True

    @property
    @abstractmethod
    def latent_spec(self) -> LatentSpec: ...

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpaceSpec: ...

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training pass.

        tokens:  (B, T, h, w) int64 -- or (B, T, D, h, w) float if continuous
        actions: (B, T-1) int64 for latent actions, (B, T-1, A) float for robot
        Returns at least `{"loss": scalar, "logits": ...}`.
        """

    @abstractmethod
    @torch.no_grad()
    def predict_next(
        self,
        tokens: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sample the next frame's latents. Returns (B, 1, h, w) / (B, 1, D, h, w).

        This is the method the RL environment calls once per `env.step`, so it
        must support incremental use (KV cache, sliding window) rather than
        re-encoding the whole history.
        """

    def reset_cache(self) -> None:
        """Drop any incremental-decoding state. Called on `env.reset`."""


class Decoder(nn.Module, ABC):
    """Stage D: latents -> pixels, optionally sharper than the tokenizer decoder.

    Optional. The tokenizer's own decoder already produces frames; a diffusion
    decoder trades ~25 extra network evaluations per frame for sharpness. On a
    8 GB card that trade is usually worth it for *visualisation* and usually not
    worth it inside an RL loop -- so `WorldModelEnv` can render with either.
    """

    @abstractmethod
    def render(self, latents: torch.Tensor, steps: int = 25, **kwargs: Any) -> torch.Tensor:
        """latents: (B, T, D, h, w) -> frames (B, T, 3, H, W) in [-1, 1]."""

    @abstractmethod
    def forward(self, frames: torch.Tensor, latents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Stage-D training pass: returns (loss, metrics)."""


__all__ = [
    "LatentSpec",
    "ActionSpaceSpec",
    "VideoTokenizer",
    "LatentActionModel",
    "Dynamics",
    "Decoder",
]
