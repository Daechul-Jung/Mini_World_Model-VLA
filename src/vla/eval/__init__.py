"""VLA evaluation.

`offline.py` works today. The simulator wrappers are scaffolds -- see
`vla/research/003_simulation_stack.md` for why SimplerEnv/LIBERO are the right
targets and plain MuJoCo is not.
"""

from .offline import evaluate_offline

__all__ = ["evaluate_offline"]
