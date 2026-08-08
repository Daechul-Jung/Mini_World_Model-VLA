# 004 -- Does a frozen 3B backbone beat a 30M model on 100 episodes?

**Status**: idea
**Slot**: none -- this is an experiment, not a component
**Depends on**: M1, M3

## Claim

Not obvious in either direction. Worth measuring before the whole research
programme is built on the assumption.

## Why now

The plan assumes "pretrained backbone + new layer" is the strong path. The
argument for it: 100 episodes cannot teach visual grounding or language
understanding, and a pretrained VLA already has both.

The argument against, which is not weak:

* Frozen features are frozen. If the representation lacks what the task needs, no
  adapter recovers it.
* OpenVLA/pi0 were trained on 7-DoF real-robot data; this dataset is 4-DoF UCSD
  pick-place. The domain gap may be larger than the pretraining advantage.
* A 30M model trained end-to-end on the target distribution can beat a frozen 3B
  model on a narrow task. This happens often enough that it should be checked.
* 4-bit quantisation degrades the features, and by an unmeasured amount.

If the pretrained path loses here, the project's whole framing changes -- better
to know in week two than in month three.

## Measurement

Same data, same head, 3 seeds, same evaluation.

| Setup | trainable params | offline `action_l1` | `gripper_transition_acc` | **sim success** |
|-------|-----------------|--------------------|-------------------------|-----------------|
| `octo_small` from scratch | ~30M | | | |
| `octo_small` frozen + adapter | ~0.4M | | | |
| pi0 frozen + adapter | ~2M | | | |
| OpenVLA frozen + adapter | ~4M | | | |
| OpenVLA `act_native` zero-shot | 0 | | | |

The second row is the controlled comparison people usually skip: freezing a
*small* model isolates "frozen vs trained" from "big vs small".

Also report data efficiency: rerun the best two at 25, 50 and 100 episodes. The
pretrained path's real claim is a better low-data slope, not a better endpoint --
and that curve is the interesting figure.

## Kill criterion

None; both outcomes are useful. A negative result here is a genuine finding about
pretrained VLAs on small out-of-distribution datasets, and it is publishable in a
way "we used a big model and it worked" is not.

## Result

*(pending)*
