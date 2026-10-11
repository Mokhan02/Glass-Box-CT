# B. Erasure (necessity) at the mid-encoder (pre-registered; results/mech/PREREG.md section B)

Rank-1 LEACE of the tumour-vs-liver concept from the stage-3 (mid) encoder features, fitted on training folds, applied on held-out folds inside the full network. Population: the 78 lesions (71 cases) the unhooked model detects (coverage >= 10%). The unhooked, identity-hooked run reproduced the stored logits (max difference 0.03 logit, fp16 storage; the first B run was invalid and discarded, DEVIATIONS 4).

| condition | change in lesion Dice | change in lesion coverage | still detected | change in liver Dice (89 cases) |
|---|---|---|---|---|
| LEACE, tumour concept | **-0.332 [-0.380, -0.286]** | -0.338 [-0.381, -0.298] | 65% [54%, 76%] | +0.002 [-0.000, +0.005] |
| random direction, same rank | -0.002 [-0.004, -0.000] | -0.002 | 100% | -0.000 |
| local-HU concept, same rank | -0.045 [-0.071, -0.019] | -0.049 | 96% | -0.023 [-0.036, -0.012] |
| LEACE minus random | **-0.330 [-0.378, -0.284]** | | | |
| LEACE minus HU-only | **-0.287 [-0.346, -0.232]** | | | |

Probe check: the tumour-vs-liver probe goes from held-out AUROC 0.958 [0.949, 0.968] to near chance after LEACE (the random and HU erasers leave it at 0.958 and 0.955; the pooled post-LEACE value 0.37 is below 0.5 because it pools five separately fitted probes, per-fold values are about 0.5). **Pre-registered verdict: USED.** Liver Dice does not change, so the erasure is not too blunt.

Plain-language summary: when the one direction that separates tumour from liver in the mid-level features is removed, the model loses about a third of the tumours it used to find and the overlap on the rest drops sharply, while its liver segmentation is unchanged. Removing a random direction of the same size does nothing, and removing a direction that only tracks local brightness does very little, so the effect is specific to the tumour-vs-liver feature. The model therefore uses this feature at the mid-encoder. Limits: this is one linear direction at one layer applied to every voxel; it shows the feature is necessary for the tumours the model currently detects, not that it is the only route; and it says nothing yet about whether adding the direction helps (C).
