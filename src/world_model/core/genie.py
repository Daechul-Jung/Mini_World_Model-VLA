"""`GenieWorldModel` -- the composition root.

Holds one instance of each contract in `base.py` and does three things:

1. Asserts the components agree (`LatentSpec` / `ActionSpaceSpec`) at construction
   time rather than at hour three of stage C.
2. `imagine()` -- autoregressive rollout, the single entry point the RL
   environment in `src/bridge/envs/world_model_env.py` uses.
3. `from_config()` / `from_checkpoints()` -- build from YAML, load per-stage
   weights, and record which checkpoint filled each slot.

It owns no layers of its own. Replacing the dynamics model with a JEPA variant
touches this file zero times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from common.checkpoint import load_component, resolve_ckpt

from .base import ActionSpaceSpec, Decoder, Dynamics, LatentActionModel, LatentSpec, VideoTokenizer
from .registry import DECODERS, DYNAMICS, LATENT_ACTIONS, TOKENIZERS


@dataclass
class RolloutResult:
    """What `imagine()` returns -- imagined steps only, context excluded."""

    frames: torch.Tensor                       # (B, n, 3, H, W) in [-1, 1]
    latents: torch.Tensor                      # (B, n, D, h, w)
    indices: Optional[torch.Tensor] = None     # (B, n, h, w) if discrete
    info: Dict[str, Any] = field(default_factory=dict)


class GenieWorldModel(nn.Module):
    """Composes tokenizer + (optional) LAM + dynamics + (optional) decoder."""

    def __init__(
        self,
        tokenizer: VideoTokenizer,
        dynamics: Dynamics,
        latent_action: Optional[LatentActionModel] = None,
        decoder: Optional[Decoder] = None,
        *,
        check_specs: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dynamics = dynamics
        self.latent_action = latent_action
        self.decoder = decoder
        self.slot_checkpoints: Dict[str, str] = {}

        if check_specs:
            tokenizer.latent_spec.assert_compatible(dynamics.latent_spec, "tokenizer/dynamics")
            if dynamics.accepts_discrete != tokenizer.latent_spec.discrete:
                raise ValueError(
                    f"{type(dynamics).__name__} expects "
                    f"{'discrete' if dynamics.accepts_discrete else 'continuous'} latents but "
                    f"{type(tokenizer).__name__} produces the other kind"
                )
            if latent_action is not None and dynamics.action_spec.kind == "latent":
                if latent_action.action_spec.num_actions != dynamics.action_spec.num_actions:
                    raise ValueError(
                        "LAM codebook size "
                        f"({latent_action.action_spec.num_actions}) != dynamics action vocab "
                        f"({dynamics.action_spec.num_actions})"
                    )

    # --------------------------------------------------------------- properties

    @property
    def latent_spec(self) -> LatentSpec:
        return self.tokenizer.latent_spec

    @property
    def action_spec(self) -> ActionSpaceSpec:
        return self.dynamics.action_spec

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def imagine(
        self,
        context_frames: torch.Tensor,
        actions: torch.Tensor | List[int] | int | None = None,
        n_steps: int = 8,
        temperature: float = 1.0,
        render: str = "tokenizer",
        decoder_steps: int = 25,
    ) -> RolloutResult:
        """Roll the world forward `n_steps` frames.

        Args:
            context_frames: (B, T_ctx, 3, H, W) in [-1, 1]. The prompt.
            actions: what to do at each imagined step.
                * `action_spec.kind == "latent"`: (B, n_steps) int64 code indices,
                  or a single int broadcast to every step.
                * `kind == "robot"`: (B, n_steps, action_dim) float. Use
                  `bridge.action_space` to produce these from a VLA.
                * `kind == "none"`: ignored.
            render: `"tokenizer"` (one forward per frame) or `"decoder"`
                (diffusion, sharper, ~`decoder_steps` forwards per frame). RL
                loops should use `"tokenizer"`; qualitative videos `"decoder"`.
        """
        was_training = self.training
        self.eval()
        try:
            device = context_frames.device
            batch = context_frames.shape[0]

            enc = self.tokenizer.encode(context_frames)
            discrete = self.latent_spec.discrete
            history = enc["indices"] if discrete else enc["latents"]

            action_seq = self._prepare_actions(actions, batch, n_steps, device)

            self.dynamics.reset_cache()
            generated: List[torch.Tensor] = []
            for step in range(n_steps):
                action = None if action_seq is None else action_seq[:, step]
                nxt = self.dynamics.predict_next(history, action=action, temperature=temperature)
                generated.append(nxt)
                history = torch.cat([history, nxt], dim=1)

            new = torch.cat(generated, dim=1)
            if discrete:
                indices = new
                latents = self.tokenizer.indices_to_latents(indices)
            else:
                indices, latents = None, new

            if render == "decoder":
                if self.decoder is None:
                    raise ValueError("render='decoder' but no stage-D decoder is loaded")
                frames = self.decoder.render(latents, steps=decoder_steps)
            else:
                frames = self.tokenizer.decode(latents)

            return RolloutResult(
                frames=frames,
                latents=latents,
                indices=indices,
                info={"render": render, "n_steps": n_steps},
            )
        finally:
            self.train(was_training)

    def _prepare_actions(
        self,
        actions: torch.Tensor | List[int] | int | None,
        batch: int,
        n_steps: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        spec = self.action_spec
        if spec.kind == "none":
            return None
        if actions is None:
            raise ValueError(f"dynamics expects {spec.kind} actions but none were given")
        if isinstance(actions, int):
            actions = torch.full((batch, n_steps), actions, dtype=torch.long, device=device)
        elif isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0).expand(batch, -1)
        actions = actions.to(device)
        if actions.shape[0] != batch:
            raise ValueError(f"action batch {actions.shape[0]} != frame batch {batch}")
        if actions.shape[1] != n_steps:
            raise ValueError(f"got {actions.shape[1]} actions for {n_steps} steps")
        return actions

    # ------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GenieWorldModel":
        """Build every slot from `cfg`. No weights are loaded."""
        tokenizer = TOKENIZERS.build(cfg["tokenizer"])
        dynamics = DYNAMICS.build(cfg["dynamics"])
        latent_action = (
            LATENT_ACTIONS.build(cfg["latent_action"]) if cfg.get("latent_action") else None
        )
        decoder = DECODERS.build(cfg["decoder"]) if cfg.get("decoder") else None
        return cls(tokenizer, dynamics, latent_action, decoder)

    @classmethod
    def from_checkpoints(
        cls,
        cfg: Dict[str, Any],
        *,
        tokenizer_ckpt: Optional[str] = None,
        dynamics_ckpt: Optional[str] = None,
        latent_action_ckpt: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        freeze: bool = True,
    ) -> "GenieWorldModel":
        """Build from config, then fill slots from per-stage checkpoints.

        Accepts `stage_a_tokenizer:best` shorthand as well as explicit paths.
        Records what filled each slot in `slot_checkpoints`, so an imagined
        rollout can always be traced back to the weights that produced it.
        """
        model = cls.from_config(cfg)
        slots = [
            ("tokenizer", tokenizer_ckpt, model.tokenizer, "tokenizer"),
            ("dynamics", dynamics_ckpt, model.dynamics, "dynamics"),
            ("latent_action", latent_action_ckpt, model.latent_action, "latent_action"),
            ("decoder", decoder_ckpt, model.decoder, "decoder"),
        ]
        for slot, spec, module, component in slots:
            if spec is None or module is None:
                continue
            path = resolve_ckpt(spec, project="world_model")
            load_component(module, path, freeze=freeze, expect_component=component)
            model.slot_checkpoints[slot] = str(path)
        return model

    def describe(self) -> str:
        def _n(m: Optional[nn.Module]) -> str:
            return "--" if m is None else f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M"

        def _t(m: Optional[nn.Module]) -> str:
            return "none" if m is None else type(m).__name__

        return (
            "GenieWorldModel\n"
            f"  tokenizer      {_t(self.tokenizer):<26} {_n(self.tokenizer)}\n"
            f"  latent_action  {_t(self.latent_action):<26} {_n(self.latent_action)}\n"
            f"  dynamics       {_t(self.dynamics):<26} {_n(self.dynamics)}\n"
            f"  decoder        {_t(self.decoder):<26} {_n(self.decoder)}\n"
            f"  latent_spec    {self.latent_spec}\n"
            f"  action_spec    {self.action_spec}\n"
            f"  checkpoints    {self.slot_checkpoints or '(none loaded)'}"
        )
