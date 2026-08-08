"""VLA stage 1 -- behaviour cloning.

Serves both training modes through one stage, because they differ only in which
parameters carry gradients:

* **From scratch** (`octo_small`, `octo_medium`): everything trainable.
* **Frozen backbone + adapter** (`openvla`, `pi0`, or Octo with
  `freeze_backbone: true`): only `modules/` and the head train.

`policy.trainable_parameters()` is what the optimiser sees, so the switch is one
config line and the loop is unchanged.

**What to expect on ~100 episodes of one task.** BC will fit the training set
quickly and the interesting number is not the training loss. Watch:

* `val/action_l1` -- generalisation to held-out episodes.
* `val/gripper_acc` -- the dimension that decides success. A policy at 95% on
  continuous dims and 60% on the gripper will fail every rollout.

Neither predicts task success. Offline action error and rollout success rate
correlate weakly, because errors concentrate at the few timesteps that matter.
Get a simulator wired up (`vla/eval/`) before trusting a BC number.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.stages import STAGES, Stage, move_to_device
from common.types import Observation
from vla.core.registry import POLICIES

from .data import build_vla_loaders


@STAGES.register("stage_bc")
class BehaviourCloningStage(Stage):
    name = "stage_bc"
    component = "policy"
    requires = ()
    monitor = "val/action_l1"
    monitor_mode = "min"

    def build(self) -> nn.Module:
        policy = POLICIES.build(self.cfg["policy"])
        if self.cfg.get("freeze_backbone", False):
            policy.freeze_backbone(True)

        spec = policy.spec
        data_cfg = self.cfg["data"]
        if data_cfg.get("obs_horizon", spec.obs_horizon) != spec.obs_horizon:
            raise ValueError(
                f"data.obs_horizon={data_cfg.get('obs_horizon')} != policy obs_horizon "
                f"{spec.obs_horizon}. These must match or the policy silently pads."
            )
        print(policy.param_summary())
        return policy

    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        train, val, self.action_spec = build_vla_loaders(
            self.cfg, self.cfg["batch_size"], self.cfg.get("seed", 0)
        )
        return train, val

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Optimise only what is actually trainable.

        A frozen 7B backbone otherwise contributes 14 GB of AdamW moment buffers
        for parameters that never change.
        """
        opt_cfg = self.cfg.get("optim", {})
        params = model.trainable_parameters()
        if not params:
            raise ValueError("nothing is trainable -- did you freeze everything?")
        return torch.optim.AdamW(
            params,
            lr=opt_cfg.get("lr", 3e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        )

    @staticmethod
    def _to_observation(batch: Dict[str, Any]) -> Observation:
        return Observation(
            image=batch["image"],
            instruction=batch.get("instruction"),
            pad_mask=batch.get("pad_mask"),
        )

    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = self._to_observation(batch)
        return model.loss(obs, batch["actions"])

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            loss, metrics = self.loss(model, batch)
            for k, v in {"loss": float(loss), **metrics}.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        model.train()

        out = {f"val/{k}": v / max(count, 1) for k, v in totals.items()}
        # Log module gates: a gate stuck at ~0 means the inserted idea is inert.
        for i, mod in enumerate(getattr(model, "modules_stack", nn.ModuleList()).mods
                                if hasattr(model, "modules_stack") else []):
            if hasattr(mod, "gate_value"):
                out[f"gate/{i}_{type(mod).__name__}"] = mod.gate_value()
        return out

    def extra_state(self) -> Dict[str, Any]:
        """Ship the action normalisation with the weights.

        Without this a checkpoint is not deployable: nothing downstream can turn
        its [-1, 1] outputs back into metres and radians.
        """
        spec = getattr(self, "action_spec", None)
        if spec is None:
            return {}
        return {
            "action_spec": {
                "dim": spec.dim,
                "q01": spec.q01,
                "q99": spec.q99,
                "mean": spec.mean,
                "std": spec.std,
                "gripper_index": spec.gripper_index,
                "name": spec.name,
            }
        }
