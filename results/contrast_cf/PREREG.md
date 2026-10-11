# Pre-registration: contrast-sign counterfactual

Written and committed BEFORE any code for this experiment exists and before any edited volume has been made or run. Choices below are fixed. Any deviation is logged in
`results/contrast_cf/DEVIATIONS.md` with its reason and timing; the experiment is reported whichever way it comes out; there is no re-analysis after seeing results.

## Hypothesis
Detection of hepatic tumours by VISTA3D class 26 depends on the SIGN of tumour-minus-liver contrast. Making a dark tumour bright lowers detection; making a bright tumour dark raises it.
Origin: an exploratory look (not pre-specified) at the sign of contrast (`results/exp8_lesions/SUMMARY.md` section 3) and a pre-specified within-patient test that supported it for 6 of 6 cases (section 4).
This experiment tests whether the dependence is causal in the image, by editing only intensities.

## Cohort and lesions
- The 89-case clean cohort: 75 UID-route cases (original conversion) + 14 geometry-route cases resolved by `scripts/convert_heldout_clean.py` (corrected volumes). All `in_distribution = False`.
- Unit = RECIST target lesion as in `scripts/recist_shift.py`: GT hepatic-tumour 3D components on annotated slices, longest axial diameter >= 10 mm, the 2 largest per case.
- Each lesion is edited ALONE in its own copy of the volume, so edits cannot interact; the outcome is measured on that lesion only.

## Measurements (identical for every volume, edited or not)
- Local liver mean L = mean HU of non-tumour GT liver (seg 1) on annotated slices, in the 5-20 mm shell around the lesion (3D distance with true spacing); if the shell has < 200 voxels, the whole non-tumour GT liver.
- Tumour mean T = mean HU of the lesion's voxels. Signed contrast c = T - L (positive = brighter than liver).
- Prediction: class 26 prompted alone, probability map resampled to the original grid exactly as in `glassbox/seg/infer.py::predict_probs`; predicted = p > 0.5.
- **Coverage** = fraction of the lesion's voxels that are predicted. **Detected** = coverage >= 0.10.
- **Lesion Dice** = Dice between the lesion and the predicted voxels lying within 10 mm (3D, true spacing) of it.
- Baseline (unedited) coverage and Dice are recomputed in this experiment with the same pipeline, once per case, not taken from earlier runs.

## Arms (each lesion appears in at most one arm; selection uses the baseline run)
- **Arm A, dark to bright:** baseline detected (coverage >= 0.10) and c < 0. Edit: set the lesion's mean HU to L + |c| (sign flips, magnitude equal).
  **Control A:** the same absolute shift in the direction that keeps the sign: set mean HU to T - 2|c| (= L - 3|c|), i.e. darker.
- **Arm B, bright to dark:** baseline coverage < 0.10 (missed or poorly detected) and c > 0. Edit: set the lesion's mean HU to L - |c|.
  **Control B:** the same absolute shift keeping the sign: set mean HU to T + 2|c| (= L + 3|c|), i.e. brighter.

## How an edit is made
- Add a constant HU offset d = (target mean - T) to voxels inside the lesion's GT mask only, with weight 1 for interior voxels and weight 0.5 for mask voxels on the 1-voxel inner border (6-connected erosion), giving a 1-voxel feather. No voxel outside the mask is changed.
- Clip edited values to [-1024, 3071] HU. The realised post-clip mean is recorded; if it differs from the target by more than 5 HU the lesion is flagged and reported, not dropped.
- Edited volumes are written as new files in a scratch directory and deleted after use; original volumes are never overwritten. Edited inputs go through the unchanged preprocessing and model.

## Outcome and statistics
For each lesion and condition: coverage and lesion Dice; change from baseline, delta = after - before.
- Arm A statistic: per lesion D_A = delta_cov(edit) - delta_cov(control A) (expected < 0). Arm B statistic: D_B = delta_cov(edit) - delta_cov(control B) (expected > 0).
- CIs: patient (case) bootstrap, 1000 resamples, 95% percentile intervals, on the median of delta_cov(edit) and on the median of D.
- Paired sign test per arm: over lesions with D != 0, the number with D < 0 (Arm A) or D > 0 (Arm B), exact two-sided binomial test against 50%.
- Lesion Dice is reported with the same statistics as a secondary outcome; the decision uses coverage.

## Decision rule (fixed)
**SUPPORTED** only if all four hold:
1. Arm A: median delta_cov(edit) < 0 with its 95% CI entirely below 0, and median D_A < 0 with its CI entirely below 0 (the edit reduces coverage beyond the sign-keeping control).
2. Arm B: median delta_cov(edit) > 0 with its CI entirely above 0, and median D_B > 0 with its CI entirely above 0.
3. Arm A paired sign test p < 0.05 in the hypothesised direction.
4. Arm B paired sign test p < 0.05 in the hypothesised direction.
**NOT SUPPORTED:** neither arm meets (1)/(2) and in both arms the point estimates of median D are zero or in the wrong direction, with at least 10 lesions per arm.
**INCONCLUSIVE:** anything else, including any arm with fewer than 10 lesions (Arm B in particular may be small).

## Overlays
Four figures, two per arm, chosen by a fixed rule: in each arm, the lesion with the largest change in coverage in the hypothesised direction (edit minus control), and the lesion with the median such change. Each shows the CT slice before and after the edit, the GT lesion, and the prediction before and after.

## Known limits, stated in advance
Editing a constant offset inside the mask changes mean contrast but not texture, edges or enhancement pattern; a real arterial-phase tumour differs in more than mean HU. A null result would mean mean contrast sign alone does not drive detection, not that phase does not.
Small Arm B (bright and missed lesions are few). Same cohort that generated the hypothesis. Lesions come from the clean cohort only; label-alignment uncertainty remains for the 14 resolved geometry cases.
