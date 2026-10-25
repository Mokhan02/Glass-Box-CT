# W3 Stage 2, M3: brightness counterfactual on models A and D (PREREG_stage2.md, as written)

Method (as pre-registered): each lesion edited alone in its own copy of the volume (`edit_lesion`: constant HU offset inside the GT mask, 0.5 weight on the inner border, clip [-1024, 3071]); prediction = the model's primary pipeline; outcome = change in lesion coverage. **Arm A** (dark to bright): baseline detected and c < 0; edit to L + |c|; sign-keeping control to T - 2|c|. **Arm B** (reverse): baseline coverage < 0.10 and c > 0; edit to L - |c|; control to T + 2|c|. D = delta_cov(edit) - delta_cov(control). Case-level bootstrap CIs (per-arm seed), paired exact sign test, at least 10 lesions per arm. Cohort: 74 primary cases (HCC_027 excluded; it contributes no M3 lesion in either model, so the including-HCC_027 sensitivity is identical). Run: A one process, D two shards (resumed after a CUDA out-of-memory, Deviation 11); all 75 D cases validated before analysis (75/75 done, 228 rows, 71 token files, 0 problems; `m3_D_validate.json`). Files: `m3/{A,D}/m3_results.json`, `m3_lesions.csv`, `m3/sens_*`.

## Verdicts (pre-registered rule)
| | A | D |
|---|---|---|
| **M3 verdict** | **SUPPORTED** | **SUPPORTED** |
| Arm A (dark to bright) n lesions / cases | 48 / 45 | 49 / 46 |
| median delta_cov(edit), 95% CI | -0.275 [-0.363, -0.141] | -0.454 [-0.595, -0.270] |
| median D_A (edit minus control) | -0.487 [-0.637, -0.292] | -0.336 [-0.542, -0.185] |
| sign test (direction as hypothesised) | 46/48, p = 8.4e-12 | 40/48 (ties dropped), p = 3.3e-06 |
| Arm B (bright to dark) n lesions / cases | 21 / 19 | 22 / 19 |
| median delta_cov(edit) | +0.124 [+0.024, +0.331] | +0.351 [+0.049, +0.583] |
| median D_B | +0.097 [+0.014, +0.345] | +0.353 [+0.058, +0.602] |
| sign test | 18/19, p = 7.6e-05 | 20/20, p = 1.9e-06 |
| secondary, lesion Dice (within 10 mm), median change of the edit: Arm A / Arm B | -0.339 / +0.208 | -0.492 / +0.452 |

Both arms are above the 10-lesion minimum in both models. Brightening a detected dark lesion removes detection and darkening a missed bright lesion restores some of it, in both models, beyond the sign-keeping controls.

## Edit accuracy (flagged edits, pre-registered: flagged, not dropped)
**10 lesions per arm per model are flagged (realised lesion mean more than 5 HU from target in the edit or its control; 20 flagged lesions per model, 40 flagged edit/control rows).** Overall median |realised - target| is 3.1 HU (A) and 3.2 HU (D), maximum 22.3 HU. The flagged edits fall short of the intended offset (for example HCC_017 rank 2: intended 115 to 71 HU, realised 89 HU), as expected when the 0.5-weight border voxels are a large share of a small lesion. The edit therefore under-delivers the intended contrast for those lesions, which would work against the effect rather than create it. **Sensitivity excluding the flagged lesions (not a pre-registered analysis; same committed code): both models stay SUPPORTED** (A: Arm A n = 38, edit -0.170 [-0.293, -0.096], D_A -0.420 [-0.616, -0.189]; Arm B n = 11, edit +0.064 [+0.013, +0.317], D_B +0.041 [+0.012, +0.320]; D: Arm A n = 39, edit -0.307 [-0.470, -0.160], D_A -0.281 [-0.440, -0.139]; Arm B n = 12, edit +0.261 [+0.066, +0.508], D_B +0.299 [+0.066, +0.524]). Arm B is down to 11 and 12 lesions in that sensitivity (still at least 10), and the effect sizes shrink, notably in A's Arm B.

## EXPLORATORY: projection onto the model's own probe direction (not part of any verdict)
Edited minus unedited mean token, projected on the cross-fitted unit-normalised probe direction of M1 (tumour vs liver), tokens inside the lesion, and its Spearman correlation with the coverage change (case bootstrap):
- **Arm A (dark to bright), EXPLORATORY:** A mean projection change -1.31, Spearman 0.76 [0.60, 0.84]; D mean projection change -0.04, Spearman 0.78 [0.64, 0.87]. In both models, lesions whose tokens move further from the tumour side of the direction after brightening lose more coverage. D's mean shift is near zero, so the correlation, not the mean, carries the reading.
- **Arm B (bright to dark), EXPLORATORY:** A mean +0.38, Spearman 0.38 [-0.11, 0.79]; D mean +0.04, Spearman 0.43 [-0.06, 0.76]. CIs include 0.

## Reading
M3 is reported beside the cross-model mechanism reading and is not part of its criterion (PREREG): **in both models the brightness edit changes detection in the hypothesised direction in both arms, beyond the sign-keeping control** (M3 SUPPORTED, A and D). This supports a causal role of lesion brightness in the image for these two models; it does not say which layer carries it (M1, M2). Limits: constant-offset edits change mean contrast only (texture, edges, size unchanged), and under-deliver for small lesions (flagged); Arm B is the small, hard arm (19 cases each); controls move HU too (T - 2|c| or T + 2|c|), so D isolates sign from magnitude only roughly; models are never pooled. The EXPLORATORY projection is a correlational bridge, not a test.
