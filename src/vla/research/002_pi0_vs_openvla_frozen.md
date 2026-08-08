# 002 -- pi0 vs OpenVLA as a frozen backbone

**Status**: idea
**Slot**: `backbones/`
**Cost**: ~1 week, mostly download and plumbing
**Depends on**: M1 (simulator)

## Claim

pi0 and OpenVLA differ mainly in action parameterisation -- flow matching vs.
discretised action tokens. Running both frozen, with the same adapter and the same
head, isolates that variable on one GPU.

## Why now

The project needs one pretrained backbone, and the choice is not obvious.

| | OpenVLA | pi0 |
|--|---------|-----|
| Params | 7B | 3.3B (PaliGemma 2.6B + 300M action expert) |
| Actions | discretised tokens over the LM vocabulary | continuous, flow matching, chunked |
| 4-bit weights | ~5-6 GB | ~2.5 GB |
| Fit on 7.7 GiB | very tight | comfortable |
| PyTorch weights | official, `trust_remote_code` | community ports (LeRobot); less settled |
| SimplerEnv/LIBERO numbers | published | fewer |

**pi0 is probably the right call on this machine**, on VRAM alone. The risk is
port quality, which is why `Pi0Policy.encode()` is left unimplemented rather than
guessed at -- it depends on which checkpoint is chosen.

## Design

Same `PolicyModule` stack, same head, same data, same schedule. The only
difference is the backbone. `encode()` returns `(B, 1, hidden_dim)` for both, so
everything downstream is literally identical code.

## Measurement

| Backbone | VRAM (load) | VRAM (train) | s/step | offline `action_l1` | **sim success** |
|----------|------------|-------------|--------|--------------------|-----------------|
| `octo_small` from scratch | | | | | |
| OpenVLA frozen + adapter | | | | | |
| pi0 frozen + adapter | | | | | |
| OpenVLA `act_native` zero-shot | | -- | | | |

`act_native` is the zero-shot reference -- OpenVLA's own action decoding, no
adapter. Any added module has to beat it.

Run `scripts/tools/vram_probe.py` before committing to either.

## Kill criterion

If neither pretrained backbone beats `octo_small` from scratch on the simulator
(idea 004), the pretrained path is not paying for its complexity on this data --
report that and focus on the from-scratch track.

## Result

*(pending)*
