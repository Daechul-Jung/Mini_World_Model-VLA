"""Long-horizon consistency: the `WorldMemory` slot.

Genie 3's headline capability is that a generated world stays consistent for
*minutes* -- walk away from a room and come back and the furniture is still
there. DeepMind attributes this to the autoregressive model attending over a long
history of past frames, with visual memory reaching roughly one minute back.

A plain causal transformer cannot do that here. At a 32x32 token grid and 10 fps,
one minute of context is 600 frames x 1024 tokens = 614k positions. So
consistency has to come from something other than "make the context longer",
and that something is this slot.

Three families worth trying, in increasing order of ambition:

1. **Compression** -- keep a bounded KV cache; summarise evicted frames into a
   small set of learned register tokens. Cheap, framework-only, no new losses.
2. **Retrieval** -- store past frame latents keyed by an estimated camera pose or
   an embedding, and retrieve the k most relevant when the view returns. This is
   the closest fit for "walk away and come back" and needs no explicit geometry.
3. **Explicit spatial state** -- maintain a persistent scene representation
   (voxel grid, 3D Gaussians, a learned map) that the dynamics model reads and
   writes. Note that DeepMind states Genie 3's consistency is *emergent*, not
   backed by an explicit 3D representation -- so this family is a deliberate
   departure from Genie, and the `splatting/` code is the natural starting point.

The contract is intentionally minimal so all three fit behind it.
See `research/007_long_horizon_memory.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class WorldMemory(nn.Module, ABC):
    """Persistent state carried across `predict_next` calls.

    A dynamics model that supports memory takes one of these and calls
    `read()` before predicting and `write()` after. A dynamics model that does
    not is unaffected -- the slot is opt-in.
    """

    @abstractmethod
    def reset(self, batch_size: int, device: torch.device) -> None:
        """Clear all state. Called on `env.reset()`."""

    @abstractmethod
    def write(self, latents: torch.Tensor, info: Optional[Dict[str, Any]] = None) -> None:
        """Record an observed or imagined frame. latents: (B, D, h, w).

        `info` may carry a pose estimate, a step index, or an action -- whatever
        the retrieval key needs. Implementations must tolerate it being absent.
        """

    @abstractmethod
    def read(self, query: torch.Tensor, k: int = 8) -> Dict[str, torch.Tensor]:
        """Return conditioning tokens for the current step.

        query: (B, D, h, w) the current frame's latents.
        Returns at least `{"tokens": (B, M, D)}` to be concatenated onto the
        dynamics model's context, plus whatever diagnostics the implementation
        wants to log (retrieval indices, attention weights).
        """

    @property
    def token_budget(self) -> int:
        """How many tokens `read` may return. The dynamics model sizes its
        position embeddings against this, so it must be a constant."""
        return 0
