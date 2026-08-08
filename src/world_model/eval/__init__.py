"""World-model evaluation.

Three questions, three modules, deliberately separate because they fail
independently:

* `recon.py`           -- can the tokenizer represent a frame?     (stage A)
* `controllability.py` -- do actions change the prediction?        (stages B+C)
* `rollout.py`         -- does a long rollout stay coherent?       (stage C, memory)

A world model can pass the first two and fail the third completely, which is
exactly the gap Genie 3's minute-long consistency claim is about.
"""

from .controllability import action_sweep, delta_psnr
from .recon import evaluate_reconstruction
from .rollout import revisit_consistency, rollout_metrics, save_rollout_video

__all__ = [
    "evaluate_reconstruction",
    "delta_psnr",
    "action_sweep",
    "rollout_metrics",
    "revisit_consistency",
    "save_rollout_video",
]
