"""OpenVLA (7B) as a frozen feature extractor with a trainable adapter + head.

**Read this before planning around OpenVLA on a 4070.**

OpenVLA is a 7B model: Prismatic VLM (Llama-2 7B + fused DINOv2/SigLIP vision) with
actions emitted as discretised tokens over the language vocabulary. Memory, at
bf16, roughly:

| Mode                          | VRAM                          | On 7.7 GiB? |
|-------------------------------|-------------------------------|-------------|
| Full fine-tune                | ~150 GB (8xA100 in the paper) | no          |
| LoRA fine-tune (r=32, bf16)   | ~27 GB (1xA100 reported)      | no          |
| 4-bit inference               | ~5-6 GB weights + activations | tight       |
| 4-bit frozen + adapter + head | ~6-7 GB                       | very tight  |

The only training route on this machine is the last row: quantise the backbone,
freeze it, and train a small `PolicyModule` stack plus an action head on top.
That is a legitimate research setup -- the standard frozen-features protocol --
and it is what this class implements. It is *not* "fine-tuning OpenVLA"; be
precise about that in any writeup.

**Be realistic about the fit.** On a 7.7 GiB laptop 4070 this leaves almost no
headroom: expect batch size 1-2, `device_map="auto"` spilling layers to CPU, and
slow steps. If it will not fit, pi0 (3.3B, ~2.5 GB at 4-bit) is the pretrained
backbone that comfortably does -- see `backbones/pi0/`. Measure before committing
a week to this path: `scripts/tools/vram_probe.py`.

**Zero-shot transfer.** OpenVLA is trained on real-robot data (Open X-Embodiment,
Bridge, RT-1). It does not transfer to arbitrary MuJoCo scenes. Evaluate it in
SimplerEnv, which was built to match the visual statistics of the Bridge and
Google-Robot datasets, or in LIBERO, for which official fine-tuned checkpoints
exist. See `vla/eval/`.

Requires `transformers`; 4-bit additionally requires `bitsandbytes`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.types import Action, Observation
from vla.core.base import PolicySpec, VLAPolicy
from vla.core.registry import HEADS, POLICIES
from vla.modules import build_modules

DEFAULT_MODEL = "openvla/openvla-7b"


@POLICIES.register(
    "openvla",
    paper="Kim et al., 2024 (arXiv:2406.09246)",
    status="frozen-backbone",
    note="7B; 4-bit frozen + adapter is the only 8 GB-feasible mode",
)
class OpenVLAPolicy(VLAPolicy):
    """Frozen OpenVLA backbone + trainable modules + trainable head.

    Args:
        model_name: HuggingFace id. `openvla/openvla-7b` is the base model;
            LIBERO-finetuned variants exist under the same org.
        load_in_4bit: quantise the backbone. Effectively required on 8 GB.
        feature_layer: which hidden layer to read features from. The last layer is
            specialised for next-token prediction over action tokens; an
            intermediate layer (`-8` or so) is often a better general
            representation. Worth ablating -- it is a one-line config change.
        freeze_backbone: keep True. Set False only on a bigger card.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_chunk: int = 1,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        feature_layer: int = -1,
        freeze_backbone: bool = True,
        image_size: int = 224,
        head: Optional[Dict[str, Any]] = None,
        modules: Optional[List[Dict[str, Any]]] = None,
        device_map: str = "auto",
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_layer = feature_layer

        from .loader import load_openvla

        self.processor, self.backbone, hidden_dim = load_openvla(
            model_name, load_in_4bit=load_in_4bit, device_map=device_map
        )
        self._hidden_dim = hidden_dim
        if freeze_backbone:
            self.freeze_backbone(True)

        self.modules_stack = build_modules(modules, hidden_dim)
        head_cfg = dict(head or {"name": "continuous_mse"})
        head_cfg.update(
            dim=self.modules_stack.out_dim, action_dim=action_dim, action_chunk=action_chunk
        )
        self.head = HEADS.build(head_cfg)

        self._spec = PolicySpec(
            action_dim=action_dim,
            action_chunk=action_chunk,
            obs_horizon=1,                     # OpenVLA is single-frame
            image_size=image_size,
            observation_keys=("image", "instruction"),
            trainable=not freeze_backbone,
            name="openvla",
        )

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    def freeze_backbone(self, freeze: bool = True) -> None:
        self.backbone.eval() if freeze else self.backbone.train()
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    # ------------------------------------------------------------------ encode

    def encode(self, obs: Observation) -> torch.Tensor:
        """-> (B, 1, hidden_dim). Pooled last-token features from `feature_layer`.

        OpenVLA consumes a single frame, so the time axis is length 1 and any
        history has to come from a `PolicyModule` that keeps its own state.
        """
        from PIL import Image

        images = obs.image[:, -1] if obs.image.ndim == 5 else obs.image
        pil = [
            Image.fromarray(
                ((img.permute(1, 2, 0).float().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
            )
            for img in images
        ]
        prompts = [
            f"In: What action should the robot take to {text.lower()}?\nOut:"
            for text in (obs.instruction or [""] * len(pil))
        ]

        inputs = self.processor(prompts, pil, return_tensors="pt").to(self.backbone.device)
        grad = torch.enable_grad() if self.spec.trainable else torch.no_grad()
        with grad:
            out = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[self.feature_layer]        # (B, L, D)
        return hidden[:, -1:].float()                        # last position, keep time axis

    def forward(self, obs: Observation) -> Action:
        features = self.modules_stack(self.encode(obs), {"observation": obs, **obs.extras})
        return Action(continuous=self.head(features), latent=features[:, -1])

    def loss(
        self, obs: Observation, target_actions: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.modules_stack(self.encode(obs), {"observation": obs, **obs.extras})
        return self.head.loss(features, target_actions, mask=mask)

    # ------------------------------------------------------- native action path

    @torch.no_grad()
    def act_native(self, obs: Observation, unnorm_key: str = "bridge_orig") -> torch.Tensor:
        """OpenVLA's own action decoding, bypassing this repo's head entirely.

        Use this for the zero-shot baseline number that any added module must
        beat. `unnorm_key` selects which dataset's action statistics the model
        de-normalises with -- getting it wrong produces actions of the right shape
        and the wrong scale, which looks like a broken policy rather than a
        configuration mistake.
        """
        from PIL import Image

        images = obs.image[:, -1] if obs.image.ndim == 5 else obs.image
        pil = Image.fromarray(
            ((images[0].permute(1, 2, 0).float().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()
        )
        prompt = f"In: What action should the robot take to {(obs.instruction or [''])[0].lower()}?\nOut:"
        inputs = self.processor(prompt, pil, return_tensors="pt").to(self.backbone.device)
        action = self.backbone.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return torch.as_tensor(action).unsqueeze(0)
