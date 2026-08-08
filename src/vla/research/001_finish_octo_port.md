# 001 -- Finish the JAX -> PyTorch Octo port

**Status**: idea (low priority)
**Slot**: `backbones/octo/`
**Cost**: 1-2 weeks
**Depends on**: a concrete need for pretrained Octo weights

## Claim

Finishing the line-by-line port would allow loading the official
`rail-berkeley/octo-small` / `octo-base` checkpoints.

## Current state

`backbones/octo/components/` and `octo_module.py` do not import. The port kept
Flax's dataclass-attribute idiom:

```python
kernel_init: Callable[...] = nn.init.xavier_uniform()   # executes at class-definition time
```

In Flax this is a field declaration. In Python it calls `xavier_uniform_()` with
no tensor and raises. There are several of these, plus `utils/typing.py` had a
`from __future__` import that was not first in the file (fixed).

## Why it is low priority

Two reasons.

1. `backbones/octo/policy.py` already implements the Octo *recipe* -- block-causal
   multimodal transformer with a readout token -- and it trains. For a
   from-scratch baseline you fully understand, that is what is wanted.
2. Loading official checkpoints needs more than "the code imports". Every
   parameter name and shape must correspond exactly to the Flax pytree, which is
   the actual work and is where the two weeks go.

The pretrained-backbone path is being served by OpenVLA/pi0 instead.

## What would make this worth doing

* A specific need for pretrained Octo weights -- e.g. Octo is the baseline a
  reviewer expects, or you want to compare adapters on Octo vs pi0 with matched
  pretraining data.
* Evidence that from-scratch Octo-small is data-limited in a way pretraining fixes
  (which idea 004 would show).

## Design, if pursued

1. Convert every Flax-style class attribute into a real `__init__` argument.
2. Write a Flax-pytree -> PyTorch `state_dict` key mapping.
3. Verify numerically: same input, same output, within fp32 tolerance, against
   the JAX model. Nothing short of this constitutes "the port works".

## Kill criterion

If step 3 cannot be made to match within a week of starting, stop. A port that
imports but produces different numbers is worse than no port -- it silently
invalidates every result built on it.

## Result

*(pending)*
