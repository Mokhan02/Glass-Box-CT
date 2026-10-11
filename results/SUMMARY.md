# Glass-Box CT: research summary (research use only; no clinical claim)

Model: NV-Segment-CT (VISTA-3D, bundle `vista3d132`), frozen. Held-out data: HCC-TACE-Seg (public, de-identified), never used for training or any fitting except where stated. All CIs are patient-level bootstrap 95% unless noted.
Status key: **PRE** = pre-registered rule, run as written; **PRE+fix** = pre-registered with a logged deviation; **DESC** = descriptive measurement, no pre-registered test; **EXPL** = exploratory.
Cohorts: *all-103* (103 cases, 28 label-alignment-unverified), *UID-75* (UID-matched route, cleanest labels), *single-acq-81* (single-acquisition clean conversion), *clean-89* (89 cases: UID-75 plus 14 re-resolved cases; counterfactual and re-fit).

**Headline.** The model represents tumours it fails to segment; its misses depend causally on the tumour being brighter than the liver; and a single threshold recalibration removes most of the one-sided RECIST under-measurement. Evidence and limits per claim are below. (Threshold shift on bright tumours: section 5, exploratory.)

## 1. What the model does wrong (hepatic tumour, class 26)  [DESC]
| number | cohort | value |
|---|---|---|
| Sum-of-diameters (RECIST-style) median bias | UID-75 | **-19% [-28%, -13%]**; all-103 pooled -24% (secondary); clean cohort -21% [-30%, -15%] |
| Detected lesions: diameter error | UID route | -11.4 mm (about -17%); zero over-estimates |
| Tumour Dice | single-acq-81 | **0.551** |
| Original readout, re-fit comparison space | clean-89 | Dice 0.555 [0.487, 0.619]; SLD median -19.2% [-29.4, -14.3]; 0% over-estimated >= 20% |
Roughly 1 in 5 lesions are largely missed (10-14% have no overlap at all). Phase cannot be labelled (acquisition-time and contrast fields are stripped from the DICOM), so no phase claim is made.
What predicts a miss [PRE, `results/exp8_lesions/SUMMARY.md`]: lesion size; nothing tested (contrast, phase proxy, window position, window content) adds to size alone, but power is low (16-29 missed lesions).

## 2. Where is the missed tissue lost? Voxel probe [PRE+fix, `results/voxel_probe/`]
Governing rule (DEVIATIONS #2): the clean cohorts decide when cohorts disagree. They do not disagree.
| cohort | verdict | encoder (mid) detection of missed interior tissue | raw-HU control | median logit of missed voxels |
|---|---|---|---|---|
| all-103 | LOST LATE | 0.587 [0.551, 0.656] | 0.114 [0.050, 0.146] | -2.59 |
| single-acq-81 | LOST LATE | 0.604 [0.571, 0.676] | 0.083 [0.041, 0.115] | -2.53 |
| UID-75 | LOST LATE | 0.627 [0.587, 0.713] | 0.112 [0.045, 0.171] | -2.40 |
**Governing verdict: LOST LATE.** The encoder carries the missed tissue; the head logits sit about 2.5 below threshold. Detection at about 5% liver false-positive rate, probe A, trained vs random-init vs raw HU (all-103; the two clean cohorts are within 0.06 of these):
| layer | trained | random-init | raw HU |
|---|---|---|---|
| mid-encoder | 0.587 [0.551, 0.656] | 0.079 [0.060, 0.090] | 0.114 [0.050, 0.146] |
| bottleneck | 0.825 [0.795, 0.869] | 0.141 [0.075, 0.178] | 0.204 [0.071, 0.281] |
| decoder_F | 0.758 [0.722, 0.821] | 0.087 [0.048, 0.110] | 0.110 [0.048, 0.141] |
Random-init control: the first run was invalid (NaN, DEVIATIONS #3); it was rerun with the logged fix (patient groups taken from the trained run). Trained CI lower bound exceeds random-init CI upper bound at every layer and cohort. Caveats: token resolution is coarse (small tumours vanish at the pooled grids: 26/145 tumour components have no token at the mid layer, 47/145 at the bottleneck); the HU control is weak (local statistics only).
Threshold recovery [PRE]: shifting the threshold recovers Dice +0.083 [+0.053, +0.116] (all-103), +0.076 [+0.038, +0.113] (UID-75).

## 3. Does contrast sign drive the misses? Counterfactual [PRE, `results/contrast_cf/`]
Edit tumour HU only inside the lesion (feathered), with sign-keeping controls. Mechanical verdict: **SUPPORTED** (both arms meet every criterion). clean-89.
- Arm A, dark tumours made bright (n=62): lesion coverage change -0.208 [-0.284, -0.098]; edit minus control -0.301 [-0.470, -0.188]; 56/62 in the hypothesised direction (sign test p < 1e-10).
- **Strong in the dark-to-bright arm (56/62); the bright-to-dark arm passes narrowly (CI lower bound +0.001).**
- Arm B, bright missed tumours made dark (n=17): +0.238 [+0.001, +0.563]; edit minus control +0.242 [+0.001, +0.515]; 13/14 non-tied in the hypothesised direction (p = 0.0018).
- 10 edits per arm missed the target contrast by more than 5 HU (flagged, kept). Interpretation: the model's detection depends on the sign of tumour-minus-liver contrast. This is a causal statement about the model, not about patients.

## 4. Can a re-fitted readout do better than a threshold shift? [PRE, `results/readout_refit/`]
Gate (pre-committed rule, `results/voxel_probe/GATE.md`): **PASSED** (governing verdict LOST LATE in both clean cohorts; trained mid CI lower > random-init CI upper in both). Re-fit run as pre-registered: 48 weights + bias, 5-fold patient-level CV, clean-89, held-out folds only.
| arm | tumour Dice | lesion detection | SLD median error | >= 20% over-estimated |
|---|---|---|---|---|
| original readout | 0.555 [0.487, 0.619] | 77.2% [67.6, 86.1] | -19.2% [-29.4, -14.3] | 0.0% |
| threshold shift, matched over-estimation | 0.635 [0.578, 0.689] | 88.1% [79.8, 95.3] | +0.6% [-4.4, 2.6] | 13.5% [6.7, 21.3] |
| re-fit readout | 0.586 [0.524, 0.647] | 86.1% [77.8, 93.1] | -1.1% [-7.1, 2.9] | 13.5% [6.7, 21.3] |
**Verdict (pre-registered rule): WORSE THAN THRESHOLD; our re-fit did not beat a single threshold shift. This does not show that no readout could do better.** Held-out Dice, re-fit minus matched threshold: -0.049 [-0.073, -0.026], at identical over-estimation (13.5% both). The re-fit does beat the original readout, but not the one-number threshold shift. For the two bright, fully-missed cases picked by a fixed rule (lowest original Dice), the re-fit recovers essentially nothing (case Dice 0.00 / 0.001, vs 0.08 for the threshold).
Caveats: the training loss (logistic + soft Dice on sampled voxels) differs from the Dice evaluated on full held-out predictions; the inner CV chose the smallest regularisation (lambda 0.001) in every fold, i.e. the grid edge; one truncated ROI (HCC_099: 34% of its tumour lies outside the predicted-liver ROI); matching uses in-sample re-fit predictions on training folds; evaluation is in the 1.5 mm preprocessed grid on annotated slices with the ROI-only readout swap (DEVIATIONS #2), so absolute numbers differ slightly from the original-geometry evaluations above.
Descriptive extras [DESC]: cosine(probe direction, original readout) = 0.48; cosine(re-fit, original) = 0.60-0.68 across folds. Mean unit-vector projection of missed tumour voxels onto the original readout: bright -1.76, dark -1.09; onto the probe direction: bright -0.79, dark -0.20. Detected voxels: original +1.04 (bright) / +1.49 (dark); probe +1.43 / +1.77. Bright-tumour Dice [EXPL]: original 0.32, threshold 0.48, re-fit 0.40 (n=31); dark: 0.68 / 0.72 / 0.68 (n=58).

## 5. Quality flag and bright-tumour threshold check
Quality flag (logistic on log predicted volume, mean probability, mean entropy; UID-75; no reference needed at run time) [DESC]: cross-validated AUROC (5-fold grouped, repeated 20x) **0.925 [0.860, 0.980]**, patient-bootstrap CI (`scripts/fit_quality_flag.py`, `glassbox/quality/flag_model.json`).
Threshold shift on bright vs dark tumours [**EXPLORATORY**, descriptive, no tests, `results/readout_refit/bright_threshold_check.md`]: the voxel-probe per-fold thresholds applied to held-out folds of the clean-89 saved logits (RECIST-target lesions, detection = coverage >= 10%; bright/dark by case-level contrast).
| group | lesions | mean coverage orig -> shifted | detected orig -> shifted | missed -> detected | detected -> missed |
|---|---|---|---|---|---|
| bright (> 0 HU) | 36 (31 cases) | 0.246 -> 0.454 | 21/36 (58%) -> 29/36 (81%) | 8 of 15 | 0 |
| dark (< 0 HU) | 65 (58 cases) | 0.593 -> 0.796 | 57/65 (88%) -> 61/65 (94%) | 4 of 8 | 0 |
Conclusion: the shift recovers bright missed tumours (8 of 15) at about the same rate as dark ones (4 of 8) and in absolute terms recovers more bright ones, since bright tumours are missed more often. It does not rescue the fully missed ones: HCC_031_552418 (+33 HU, 43 and 38 mm) and HCC_078_958325 (+19 HU, 40 mm) reach coverage 0.00-0.06, still below the 10% detection line. Small numbers; read as descriptive only.

## What can and cannot be said
- Can: the model under-measures hepatic tumour diameters by about a fifth; the missed tissue is represented in the encoder and lost downstream; the model's detection depends on contrast sign (causally, by editing); a single threshold shift removes most of the bias on this cohort, and our 48-weight re-fit did not beat it (this does not exclude a better readout).
- Cannot: anything about patients or outcomes; anything about phase; that the threshold shift generalises beyond HCC-TACE-Seg (it was chosen on HCC cases only); that the bias is the same on other cohorts (the research in this file is HCC-TACE-Seg only).
- Pre-registration record: PREREG.md / DEVIATIONS.md in each results folder, committed before the corresponding results.
