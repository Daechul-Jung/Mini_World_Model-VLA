"""Octo-style PyTorch policy -- trainable from scratch on an 8 GB card.

Follows the Octo recipe (Team et al., 2024) rather than porting it line by line:

    per timestep:  [ image patch tokens | language tokens | readout token ]
    across time:   causal transformer with block-wise attention
    readout token -> features -> modules -> action head

The block-attention rule matters and is the part people get wrong: observation
tokens at step t may attend to observations at steps <= t, and the *readout*
token attends to everything up to t but nothing attends back to it. That keeps
readouts from leaking into the observation representation, which is what lets you
add a second head later without retraining the trunk.

**Sizes.** Octo's own ladder is Small 27M / Base 93M. `octo_small` here is
comparable to Octo-Small; `octo_medium` sits between the two. Neither will match
the published checkpoints: Octo was trained on 800k trajectories from 25 datasets
on TPU pods, and this repo has ~100 episodes of one task. Treat from-scratch
training as *learning the architecture and the pipeline*, and expect real
manipulation competence to come from the pretrained-backbone path instead.

**Language.** Octo uses a frozen T5-base encoder. That is available here via
`language_encoder="t5-small"` (needs `transformers`), but the default is a
learned hash-embedding bag of words, which has no dependency and is honestly
sufficient for a 3-instruction dataset. Switch to T5 when instructions get
compositional -- that is what the frozen encoder buys.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.types import Action, Observation
from vla.core.base import PolicySpec, VLAPolicy
from vla.core.registry import HEADS, POLICIES
from vla.modules import build_modules


# --------------------------------------------------------------------- encoders


class PatchImageEncoder(nn.Module):
    """Small conv stem + patch projection. Cheaper than a ViT at 128-256 px."""

    def __init__(self, dim: int = 384, patch: int = 16, in_channels: int = 3, image_size: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.proj = nn.Conv2d(128, dim, patch // 4, stride=patch // 4)
        self.n_tokens = (image_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B*T, 3, H, W) -> (B*T, n_tokens, dim)."""
        z = self.proj(self.stem(images))
        return z.flatten(2).transpose(1, 2) + self.pos


class HashBagOfWords(nn.Module):
    """Dependency-free instruction encoder: hashed word embeddings, mean-pooled.

    Adequate when the instruction set is small and closed (this repo's OpenX
    subset has a handful of phrasings). It cannot generalise to unseen wordings
    the way a pretrained text encoder can -- that is the trade, and the reason
    `language_encoder="t5-small"` exists.
    """

    def __init__(self, dim: int = 384, vocab: int = 8192, n_tokens: int = 4):
        super().__init__()
        self.vocab = vocab
        self.n_tokens = n_tokens
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, n_tokens * dim)
        self.dim = dim

    def _hash(self, words: List[str], device: torch.device) -> torch.Tensor:
        ids = [int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % self.vocab for w in words] or [0]
        return torch.tensor(ids, device=device)

    def forward(self, instructions: List[str], device: torch.device) -> torch.Tensor:
        """-> (B, n_tokens, dim)."""
        pooled = torch.stack(
            [self.embed(self._hash(text.lower().split(), device)).mean(0) for text in instructions]
        )
        return self.out(pooled).reshape(len(instructions), self.n_tokens, self.dim)


class FrozenT5(nn.Module):
    """Frozen T5 encoder, matching Octo's language conditioning."""

    def __init__(self, dim: int, model_name: str = "t5-small", max_length: int = 16):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(self.encoder.config.d_model, dim)
        self.max_length = max_length
        self.n_tokens = max_length

    def forward(self, instructions: List[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.encoder(**batch).last_hidden_state
        return self.proj(out)


# ------------------------------------------------------------------- transformer


class BlockAttentionLayer(nn.Module):
    """One transformer layer using a precomputed additive attention mask."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


# ------------------------------------------------------------------------ policy


@POLICIES.register(
    "octo_torch",
    paper="Team et al., 2024 (octo-models.github.io)",
    status="baseline",
    note="from-scratch Octo-style transformer; trainable on 8 GB",
)
class OctoTorchPolicy(VLAPolicy):
    """Octo-style multimodal transformer policy.

    Args:
        action_dim: robot action dimension. **Set this from the dataset**, not by
            hand -- UCSD pick-place is 4-DoF, Bridge is 7-DoF, and a mismatch
            here trains a model that cannot be evaluated anywhere else.
        action_chunk: actions predicted per step. Octo predicts 4; executing only
            the first (receding horizon) is usually better than executing all.
        obs_horizon: frames of history the transformer sees.
        head: registry config for the action head.
        modules: list of `PolicyModule` configs inserted before the head.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_chunk: int = 4,
        obs_horizon: int = 2,
        image_size: int = 256,
        patch: int = 16,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        dropout: float = 0.0,
        language_encoder: str = "hash",
        language_model: str = "t5-small",
        use_wrist: bool = False,
        head: Optional[Dict[str, Any]] = None,
        modules: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.obs_horizon = obs_horizon
        self.use_wrist = use_wrist

        self.image_encoder = PatchImageEncoder(dim, patch, 3, image_size)
        self.wrist_encoder = PatchImageEncoder(dim, patch, 3, image_size) if use_wrist else None
        self.language = (
            FrozenT5(dim, language_model) if language_encoder == "t5" else HashBagOfWords(dim)
        )

        self.n_img = self.image_encoder.n_tokens * (2 if use_wrist else 1)
        self.n_lang = self.language.n_tokens
        self.tokens_per_step = self.n_img + self.n_lang + 1          # +1 readout
        self.readout = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.readout, std=0.02)

        self.time_embed = nn.Parameter(torch.zeros(1, obs_horizon, 1, dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 3, dim))       # image / lang / readout
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        self.layers = nn.ModuleList(
            [BlockAttentionLayer(dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("attn_mask", self._build_mask(obs_horizon), persistent=False)

        self.modules_stack = build_modules(modules, dim)
        head_cfg = dict(head or {"name": "continuous_mse"})
        head_cfg.update(dim=self.modules_stack.out_dim, action_dim=action_dim, action_chunk=action_chunk)
        self.head = HEADS.build(head_cfg)

        self._spec = PolicySpec(
            action_dim=action_dim,
            action_chunk=action_chunk,
            obs_horizon=obs_horizon,
            image_size=image_size,
            observation_keys=("image", "instruction") + (("wrist_image",) if use_wrist else ()),
            name="octo_torch",
        )

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    # -------------------------------------------------------------------- mask

    def _build_mask(self, horizon: int) -> torch.Tensor:
        """Octo's block-causal rule as an additive float mask.

        * Observation/language tokens at step t see all obs tokens at steps <= t.
        * The readout token at step t sees obs tokens at steps <= t and itself.
        * Nothing attends *to* a readout token -- so heads stay detachable.
        """
        n = self.tokens_per_step
        total = horizon * n
        mask = torch.full((total, total), float("-inf"))

        readout_positions = {t * n + n - 1 for t in range(horizon)}
        for i in range(total):
            t_i = i // n
            for j in range(total):
                t_j = j // n
                if j in readout_positions and j != i:
                    continue                          # never attend to another readout
                if t_j <= t_i:
                    mask[i, j] = 0.0
        return mask

    # ------------------------------------------------------------------ encode

    def encode(self, obs: Observation) -> torch.Tensor:
        """-> (B, T, D) readout features, one per timestep."""
        images = obs.image
        if images.ndim == 4:                         # (B, 3, H, W) -> add time
            images = images.unsqueeze(1)
        b, t = images.shape[:2]
        if t != self.obs_horizon:
            images = self._fit_horizon(images)
            t = self.obs_horizon

        img_tokens = self.image_encoder(images.flatten(0, 1)).unflatten(0, (b, t))
        if self.use_wrist and obs.wrist_image is not None:
            wrist = obs.wrist_image
            if wrist.ndim == 4:
                wrist = wrist.unsqueeze(1)
            wrist_tokens = self.wrist_encoder(
                self._fit_horizon(wrist).flatten(0, 1)
            ).unflatten(0, (b, t))
            img_tokens = torch.cat([img_tokens, wrist_tokens], dim=2)

        instructions = obs.instruction or [""] * b
        lang = self.language(instructions, images.device).unsqueeze(1).expand(b, t, -1, -1)

        readout = self.readout.expand(b, t, 1, self.dim)

        x = torch.cat(
            [
                img_tokens + self.type_embed[:, 0],
                lang + self.type_embed[:, 1],
                readout + self.type_embed[:, 2],
            ],
            dim=2,
        )
        x = (x + self.time_embed[:, :t]).reshape(b, t * self.tokens_per_step, self.dim)

        mask = self.attn_mask
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x).reshape(b, t, self.tokens_per_step, self.dim)
        return x[:, :, -1]                            # readout token per timestep

    def _fit_horizon(self, images: torch.Tensor) -> torch.Tensor:
        """Pad by repeating the first frame, or keep the most recent frames."""
        t = images.shape[1]
        if t == self.obs_horizon:
            return images
        if t > self.obs_horizon:
            return images[:, -self.obs_horizon :]
        pad = images[:, :1].expand(-1, self.obs_horizon - t, -1, -1, -1)
        return torch.cat([pad, images], dim=1)

    # ------------------------------------------------------------------ forward

    def forward(self, obs: Observation) -> Action:
        features = self.encode(obs)
        context = {"observation": obs, **obs.extras}
        features = self.modules_stack(features, context)
        return Action(continuous=self.head(features), latent=features[:, -1])

    def loss(
        self, obs: Observation, target_actions: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.encode(obs)
        features = self.modules_stack(features, {"observation": obs, **obs.extras})
        return self.head.loss(features, target_actions, mask=mask)

    def sample(self, obs: Observation, temperature: float = 1.0) -> Action:
        """Stochastic action + log-prob, for RL post-training."""
        if not self.head.supports_rl:
            raise RuntimeError(
                f"{type(self.head).__name__} has no action distribution. RL "
                "post-training needs head.name=gaussian or discrete_bins."
            )
        features = self.encode(obs)
        features = self.modules_stack(features, {"observation": obs, **obs.extras})
        action, logp = self.head.sample(features, temperature)
        return Action(continuous=action, logp=logp, latent=features[:, -1])

    # ----------------------------------------------------------------- freezing

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the trunk and image/language encoders; leave modules + head free.

        This is the same switch the large-backbone policies expose, so a training
        config written for OpenVLA works unchanged here.
        """
        for part in (self.image_encoder, self.wrist_encoder, self.language, self.layers, self.norm):
            if part is None:
                continue
            for p in part.parameters():
                p.requires_grad_(not freeze)
        for tensor in (self.readout, self.time_embed, self.type_embed):
            tensor.requires_grad_(not freeze)


@POLICIES.register("octo_small", status="baseline", note="~30M params, 128px")
def octo_small(**kwargs: Any) -> OctoTorchPolicy:
    """Comparable to Octo-Small (27M). The size to start from on a 4070."""
    defaults = dict(dim=384, depth=12, num_heads=6, image_size=128, patch=16, obs_horizon=2)
    defaults.update(kwargs)
    return OctoTorchPolicy(**defaults)


@POLICIES.register("octo_medium", status="baseline", note="~90M params, 256px")
def octo_medium(**kwargs: Any) -> OctoTorchPolicy:
    """Between Octo-Small and Octo-Base. Fits 8 GB at batch 8 with bf16 + grad accum.

    Do not start here. Get `octo_small` learning on the pick-and-place subset
    first -- with ~100 episodes the bottleneck is data, and a bigger trunk mostly
    buys a faster route to overfitting.
    """
    defaults = dict(dim=768, depth=12, num_heads=12, image_size=256, patch=16, obs_horizon=2)
    defaults.update(kwargs)
    return OctoTorchPolicy(**defaults)
