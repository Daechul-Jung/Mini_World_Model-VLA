# 011 -- One dynamics model, latent AND robot actions

**Status**: idea
**Slot**: `dynamics/`
**Cost**: ~3 days
**Depends on**: W8

## Claim

A dynamics model conditioned on the concatenation of a latent action code and a
robot action vector, trained with modality dropout, can learn from *both*
unlabelled room video and labelled robot video -- and can then be driven by a VLA
exactly, with no translation layer.

## Why now

This project has two kinds of video and they are currently mutually exclusive:

| Data | Action labels | Currently usable by |
|------|--------------|--------------------|
| TUM RGB-D, LSUN, phone video | none | `action_kind=latent` only |
| OpenX episodes | 4-DoF robot | `action_kind=robot` only |

So the latent-action path gets the visual variety and cannot be driven exactly,
and the robot path can be driven exactly and is starved of data. Unifying them
gets both.

This is the most interesting unexplored idea in the world-model track, and it is
the clean solution to the bridge's central problem rather than a workaround for
it.

## Design

```
action_embedding = W_latent · embed(a_latent) + W_robot · proj(a_robot)
```

with independent dropout on each term during training:

* unlabelled clip -> latent code present, robot action dropped
* labelled clip   -> both present (the LAM still labels it)
* at rollout      -> supply either, or both

`ActionSpaceSpec` needs a `kind="both"` variant. `GenieWorldModel`'s
compatibility check needs to accept it. Nothing else changes.

The interesting hypothesis beyond data efficiency: with both present on labelled
clips, the model is forced to discover *which part* of the latent code is
redundant with the robot action -- and the residual is exactly the uncontrollable
part of the scene. That is a potentially useful disentanglement, and it is
measurable via the mutual information between the two conditioning paths.

## Measurement

| Model | trained on | `delta_psnr` (latent) | robot-action fidelity | rollout PSNR |
|-------|-----------|----------------------|----------------------|--------------|
| latent-only | TUM | | n/a | |
| robot-only | OpenX | n/a | | |
| unified | TUM + OpenX | | | |

"Robot-action fidelity": predict frame t+1 under the true robot action vs. a
perturbed one, and take the PSNR gap -- the robot-action analogue of Delta-PSNR.

## Kill criterion

If the unified model is worse than robot-only on robot-action fidelity, the extra
data is not transferring across the domain gap (desk video vs. robot tabletop) --
which would itself be a clean, reportable negative result about domain transfer
in world models.

## Result

*(pending)*
