"""Octo backbone.

Two things live here:

* `policy.py` -- **the working implementation.** An Octo-style transformer written
  directly in PyTorch, registered as `octo_torch` / `octo_small` / `octo_medium`.
  Use this.

* `octo_module.py`, `octo_model.py`, `components/`, `utils/` -- an in-progress
  line-by-line port of the upstream JAX/Flax Octo (see `UPSTREAM.md`). It does
  **not** import cleanly: the port kept Flax's dataclass-attribute style, so
  class bodies like `kernel_init: Callable = nn.init.xavier_uniform()` execute at
  class-definition time and raise. Finishing it is worthwhile only if the goal is
  loading the official `rail-berkeley/octo-*` checkpoints, which requires
  matching parameter names exactly.

  It is deliberately not imported here, so a broken port never breaks the
  package. `vla/research/001_finish_octo_port.md` tracks the work.
"""

from .policy import OctoTorchPolicy, octo_medium, octo_small

__all__ = ["OctoTorchPolicy", "octo_small", "octo_medium"]
