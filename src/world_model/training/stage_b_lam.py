"""Stage B -- train the latent action model.

**Deviation from the paper, on purpose.** Genie trains the video tokenizer first,
then *co-trains* the LAM and the dynamics model. This project trains the LAM as
its own stage because the whole point of the repo layout is that every component
gets an independently inspectable checkpoint on an 8 GB card. The cost is real:
co-training lets the dynamics model's needs shape which transitions the LAM calls
distinct, and a standalone LAM optimises only its own next-frame reconstruction.

`configs/world_model/stage_c_dynamics.yaml` therefore exposes
`latent_action.freeze: false`, which continues training the LAM inside stage C
starting from this checkpoint -- Genie's co-training, reachable as a second phase
rather than the only option. Run stage B standalone first regardless: it is the
cheapest place to discover that your clips have no motion in them.

**Requires video.** LSUN room stills cannot train this stage -- there is no
"next frame". Use TUM RGB-D sequences, extracted phone video, or OpenX episodes.

**The metric that matters** is `action_perplexity`, not reconstruction. If it
collapses toward 1, every transition is being assigned the same code, the action
channel is carrying nothing, and the stage-C dynamics model will learn to ignore
actions completely -- which looks fine in loss curves and produces a world model
you cannot steer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.stages import STAGES, Stage, move_to_device
from world_model.core.registry import LATENT_ACTIONS

from .data import build_loaders


@STAGES.register("stage_b_latent_action")
class LatentActionStage(Stage):
    name = "stage_b_latent_action"
    component = "latent_action"
    requires = ()          # pixel input: the stage-A tokenizer is NOT needed
    monitor = "val/loss"
    monitor_mode = "min"

    def build(self) -> nn.Module:
        lam_cfg = dict(self.cfg["latent_action"])
        # `freeze` is a stage-C directive (co-train the LAM or not), not a
        # constructor argument. One config block serves both stages.
        lam_cfg.pop("freeze", None)
        model = LATENT_ACTIONS.build(lam_cfg)
        clip_len = self.cfg["data"].get("clip_len", 2)
        if clip_len < 2:
            raise ValueError("stage B needs clip_len >= 2; got " + str(clip_len))
        return model

    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        return build_loaders(self.cfg, self.cfg["batch_size"], self.cfg.get("seed", 0))

    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, loss, metrics = model(batch["frames"])
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        metrics = super().evaluate(model, loader)
        used = metrics.get("val/actions_used", 0.0)
        total = model.action_spec.num_actions or 1
        if used <= 1.5:
            # Loud, because this failure is invisible until stage C rollouts.
            print(
                f"  [stage B WARNING] only {used:.1f}/{total} action codes in use. "
                "The bottleneck has collapsed -- the dynamics model will not be "
                "steerable. Try: more motion between frames (raise data.frame_skip), "
                "a smaller num_actions, or a lower VQ commitment weight."
            )
        return metrics

    @torch.no_grad()
    def on_epoch_end(self, model: nn.Module, epoch: int, metrics: Dict[str, float]) -> None:
        """Log the action-code histogram so collapse is visible in TensorBoard."""
        return None
