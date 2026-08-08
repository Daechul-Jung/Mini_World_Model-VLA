"""Stage C -- train the dynamics model on frozen tokens.

This is where the world model actually learns "what happens next". Inputs:

* frozen stage-A tokenizer -> token clips (no gradient, `torch.no_grad`)
* actions, from one of three sources depending on `dynamics.action_kind`:
    - `latent`: the stage-B LAM labels each transition (frozen by default;
      set `latent_action.freeze: false` for Genie-style co-training)
    - `robot`:  the dataset's own action labels (OpenX episodes)
    - `none`:   unconditional next-frame prediction, useful as a smoke test

**Watch for the degenerate solution.** Next-token cross-entropy on video is
minimised well by copying the previous frame, because most tokens do not change.
`token_acc` will look excellent and the model will be useless. The diagnostic is
`copy_baseline_acc` -- the accuracy of literally predicting frame t for frame
t+1, logged alongside. If the model is not clearly beating it, nothing has been
learned about dynamics.

**Controllability** (`val/delta_psnr`) is the Genie metric and the one that says
whether actions matter: roll the model forward under the true latent action and
under a random one, and compare PSNR against ground truth. A model that ignores
actions scores ~0 here regardless of its loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.checkpoint import load_component, resolve_ckpt
from common.stages import STAGES, Stage, move_to_device
from world_model.core.registry import DYNAMICS, LATENT_ACTIONS, TOKENIZERS

from .data import build_loaders


@STAGES.register("stage_c_dynamics")
class DynamicsStage(Stage):
    name = "stage_c_dynamics"
    component = "dynamics"
    requires = ("stage_a_tokenizer",)
    monitor = "val/loss"
    monitor_mode = "min"

    def build(self) -> nn.Module:
        # --- frozen tokenizer -------------------------------------------------
        self.tokenizer = TOKENIZERS.build(self.cfg["tokenizer"]).to(self.device)
        tok_ckpt = self.ctx.parent_ckpts.get("tokenizer")
        if tok_ckpt is None:
            raise ValueError(
                "stage C needs --tokenizer_ckpt (or stage_a_tokenizer:best). "
                "Training dynamics on an untrained tokenizer produces tokens with "
                "no stable meaning."
            )
        load_component(
            self.tokenizer, resolve_ckpt(tok_ckpt), freeze=True, expect_component="tokenizer"
        )

        # --- action source ----------------------------------------------------
        dyn_cfg = dict(self.cfg["dynamics"])
        spec = self.tokenizer.latent_spec
        dyn_cfg.setdefault("latent_grid", spec.grid)
        dyn_cfg.setdefault("vocab_size", spec.vocab_size)
        dyn_cfg.setdefault("latent_dim", spec.dim)
        self.action_kind = dyn_cfg.get("action_kind", "latent")

        self.lam: Optional[nn.Module] = None
        self.lam_trainable = False
        if self.action_kind == "latent":
            lam_ckpt = self.ctx.parent_ckpts.get("latent_action")
            if lam_ckpt is None:
                raise ValueError(
                    "action_kind='latent' needs --latent_action_ckpt from stage B"
                )
            lam_cfg = dict(self.cfg["latent_action"])
            self.lam_trainable = not lam_cfg.pop("freeze", True)
            self.lam = LATENT_ACTIONS.build(lam_cfg).to(self.device)
            load_component(
                self.lam,
                resolve_ckpt(lam_ckpt),
                freeze=not self.lam_trainable,
                expect_component="latent_action",
            )

        model = DYNAMICS.build(dyn_cfg)
        if self.lam_trainable:
            # Genie-style co-training: the LAM's parameters join the optimiser by
            # being registered as a child of the trained module.
            model.co_trained_lam = self.lam
        return model

    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        return build_loaders(self.cfg, self.cfg["batch_size"], self.cfg.get("seed", 0))

    # ------------------------------------------------------------------ tokens

    def _tokens_and_actions(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        frames = batch["frames"]
        with torch.no_grad():
            tokens = self.tokenizer.encode(frames)["indices"]      # (B, T, h, w)

        extra: Dict[str, float] = {}
        if self.action_kind == "none":
            return tokens, None, extra
        if self.action_kind == "robot":
            if "actions" not in batch:
                raise KeyError(
                    "action_kind='robot' but the dataset has no 'actions'. Use "
                    "data.name=episode_npz (OpenX), which is the only source here "
                    "with labelled robot actions."
                )
            return tokens, batch["actions"], extra

        ctx = torch.enable_grad() if self.lam_trainable else torch.no_grad()
        with ctx:
            inferred = self.lam.infer_actions(frames)
        actions = inferred["indices"]
        if self.lam_trainable:
            extra["lam_vq_loss"] = float(inferred["aux_loss"])
        return tokens, actions, extra

    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        tokens, actions, extra = self._tokens_and_actions(batch)
        out = model(tokens, actions)
        loss = out["loss"]

        metrics = {"token_acc": float(out["token_acc"]), **extra}
        with torch.no_grad():
            flat = tokens.flatten(2)
            metrics["copy_baseline_acc"] = float((flat[:, :-1] == flat[:, 1:]).float().mean())
        return loss, metrics

    # -------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        metrics = super().evaluate(model, loader)
        if self.action_kind == "latent":
            metrics["val/delta_psnr"] = self._controllability(model, loader)
        return metrics

    @torch.no_grad()
    def _controllability(self, model: nn.Module, loader: DataLoader, max_batches: int = 4) -> float:
        """Genie's Delta-PSNR: does conditioning on the *right* action help?

        For each clip, predict frame t+1 from the true latent action and from a
        random one, decode both, and take PSNR(true) - PSNR(random). A model that
        ignores its action input scores ~0 no matter how low its loss is.
        """
        from common.metrics import psnr

        model.eval()
        deltas = []
        num_actions = self.lam.action_spec.num_actions
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            frames = batch["frames"]
            tokens = self.tokenizer.encode(frames)["indices"]
            true_actions = self.lam.infer_actions(frames)["indices"]

            ctx, target = tokens[:, :-1], frames[:, -1:]
            a_true = true_actions[:, -1]
            a_rand = torch.randint_like(a_true, num_actions)

            pred_true = model.predict_next(ctx, action=a_true, temperature=1.0)
            pred_rand = model.predict_next(ctx, action=a_rand, temperature=1.0)
            img_true = self.tokenizer.decode_indices(pred_true)
            img_rand = self.tokenizer.decode_indices(pred_rand)
            deltas.append(psnr(img_true, target) - psnr(img_rand, target))
        model.train()
        return sum(deltas) / max(len(deltas), 1)

    def extra_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"action_kind": self.action_kind}
        if self.lam_trainable and self.lam is not None:
            state["latent_action"] = self.lam.state_dict()
        return state
