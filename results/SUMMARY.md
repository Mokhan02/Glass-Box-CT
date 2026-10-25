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

## 6. Mechanistic follow-up (`results/mech/`; clean-89, 5-fold patient-level CV, patient-bootstrap CIs; rules in `results/mech/PREREG.md`, deviations in `results/mech/DEVIATIONS.md`)
| exp | question | result | status |
|---|---|---|---|
| A, direct logit attribution | which decoder_F channels carry the missed-vs-detected gap? | 8 channels (36, 20, 10, 22, 18, 30, 17, 7) explain **0.77 [0.76, 0.78]** of the gap on held-out cases: **clearly concentrated**. Same channels carry the bright-vs-dark gap (76% [71%, 83%] in the same 8; Spearman -0.95 [-0.98, -0.85], opposite sign), but the pre-registered rule required a positive correlation, so its literal verdict is **"no"** (DEVIATIONS 1). cos(readout, probe direction) = 0.35. | pre-registered; sign-aware reading post hoc |
| A, stratified | does brightness lower those channels within voxels of the same outcome? | Yes: detected voxels -0.90 [-1.48, -0.34] logit (83% in the same 8), missed voxels -1.45 [-1.97, -0.96] (73%). | **exploratory**, post hoc |
| D, patching | where does the dark-to-bright coverage loss arise? | Dark mid-encoder features patched into the bright run inside the tumour + 10 mm restore **0.79 [0.71, 0.85]** of the coverage drop (bottleneck 0.66 [0.55, 0.76]; decoder_F 1.0 by construction); shell-only 0.06, far context 0.03 at mid; unrelated-case control negative. Earliest layer >= 50%: mid (the earliest tested). 32 eligible lesions. Shell-only at the bottleneck uninformative (DEVIATIONS 3). | pre-registered |
| B, erasure | is the tumour-vs-liver direction used at the mid-encoder? | LEACE: lesion Dice **-0.33 [-0.38, -0.29]** on the 78 lesions the model detects (65% still detected); random direction -0.002, local-HU direction -0.045; liver Dice unchanged (+0.002). Verdict **USED**. | pre-registered (first run invalid and discarded, DEVIATIONS 4) |
| bridge | does brightness lower that direction in the same tumour? | Projection bright minus dark **-0.93 [-1.22, -0.66]** (55 of 62 lower, p = 2e-10); correlation of projection drop with coverage drop Spearman 0.74 [0.58, 0.85] (eligible: 0.62 [0.33, 0.81]). | **exploratory**, post hoc |
| C, steering | does adding the direction recover missed tumours better than a threshold shift? | Probe steering beats random directions (+0.020 Dice) but not the matched threshold shift (mid -0.006 [-0.013, +0.001]; bottleneck -0.003 [-0.009, +0.002]): **EQUIVALENT TO THRESHOLD** at both layers. Alpha sat at the grid edge in every fold, so stronger steering was not tested; matching on false-positive volume makes the comparator a weak threshold shift. | pre-registered |
One-line reading: the model's detection depends on a tumour-vs-liver feature at the mid-encoder; making a tumour brighter lowers that feature inside the tumour (exploratory bridge), removing it removes detection (B), restoring it restores detection (D), but adding it back behaves like a threshold shift, not a better detector (C). All of this is about the model, not about patients.

## 7. Trust layer (`results/trust/`; clean-89; rules in `results/trust/PREREG.md`, written before these analyses were run)
**Flag-gated recalibration [PRE, CPU on the saved logits].** Apply the recalibrated threshold (logit -3.0) only when the quality flag (refitted inside each training fold) predicts under-segmentation; otherwise keep the default. 5-fold patient-level CV, 89 cases with a measurable target, 49 gated to the recalibrated threshold.
| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |
|---|---|---|---|---|---|
| default | -19.2% | 34.1% | 0.0% | 49.4% | 0.555 |
| recalibrated | -0.1% | 21.9% | 12.4% | 31.5% | 0.640 |
| gated | -7.7% | 20.6% | 3.4% | 29.2% | 0.647 |
Pre-registered decision: **SHIPPED** (the gated policy beats both). Gated minus always-recalibrated, over-estimation indicator -0.090 (95% CI -0.157 to -0.034); gated minus always-default, absolute sum-of-diameters error -0.135 (95% CI -0.195 to -0.080). Caveats: the recalibrated threshold and the flag's operating point were chosen on these same cases (cross-validated, but expect less on new data); the gated policy still leaves a median under-measurement of 7.7%; about 89 cases.
**Sensitivity analysis [NOT pre-registered; nested CV].** The recalibrated threshold (grid -1.0 to -4.0 logit) and the flag's sensitivity target (0.80 to 0.95) are chosen inside each outer training fold only (inner 4-fold CV of the gated policy's mean absolute error), then applied to the held-out fold. 95% patient-bootstrap CIs in brackets.
| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |
|---|---|---|---|---|---|
| default (nested) | -19.2% [-28.9, -14.3] | 34.1% [26.7, 41.8] | 0.0% | 49.4% | 0.555 |
| recalibrated (nested) | -0.1% [-6.3, +2.7] | 21.9% [16.3, 28.5] | 12.4% [5.6, 20.2] | 31.5% | 0.640 |
| gated (nested) | -4.5% [-6.2, -2.0] | 19.8% [15.2, 25.2] | 7.9% [3.4, 14.6] | 31.5% | 0.645 |
Gated minus always-recalibrated, over-estimation indicator -0.045 (95% CI -0.124 to +0.034); gated minus always-default, absolute error -0.143 (95% CI -0.202 to -0.085). **Applied to this analysis, the pre-registered shipping rule is NOT met** (the over-estimation interval includes zero). Chosen thresholds: -3.5 logit for the gated policy in all five folds, -3.0 for always-recalibrated. **Quoted in the app and README: the nested analysis** (the stricter test, with the less favourable over-estimation rate); the reason the quoting rule was changed after both results were seen is in `results/trust/DEVIATIONS.md` #1. Honest reading: the gated policy still cuts the sum-of-diameters error from 34.1% to about 20% and recovers most of the Dice, but whether it over-estimates less than always-recalibrated is not established.

**Contrast-phase heuristic [PRE rule; descriptive evaluation; `results/trust/PHASE_RESULTS.md`]: UNVALIDATED, kept in the audit record only.** Phase estimated from the model's own aorta and portal-vein HU with literature thresholds fixed before the check (aorta < 100 HU unenhanced; >= 200 HU arterial-like). Estimates on the 89 cases: arterial-like 21, venous-like 65, unenhanced 3. Agreement of arterial-like with the reference-contrast sign (reference > 0 HU): 60% (95% CI 49% to 70%), kappa 0.04 (CI -0.16 to 0.24), i.e. no better than chance. Largest reference lesion missed (default coverage < 10%): 28% (11% to 46%) when the blind-spot warning fires vs 17% (8% to 28%) when quiet, a difference of +11 points (95% CI -9 to +31): in the expected direction but not statistically clear. **Product decision (2026-10-12, after these results): the phase estimate is removed from verdicts, reasons and the main UI; the bright-tumour warning uses the measured tumour-minus-liver contrast only; the estimate is kept in the audit record, labelled "unvalidated heuristic (agreement with reference contrast sign at chance level, kappa 0.04)".** All five demo scans are estimated venous-like, so none exercised the arterial branch.

**Audit record and verdicts [product features, no new numbers].** Each scan and lesion gets a Trusted / Review / Unreliable verdict from transparent rules (`glassbox/trust/verdict.py`, not tuned on the demo scans); each scan has a downloadable JSON and PDF audit record with the model checkpoint hash, thresholds, phase, flag, default vs recalibrated values, the agent report, the tool-call log and an integrity checksum.

**Two-timepoint check [SIMULATED, assumption-dependent; `results/trust/LONGITUDINAL.md`].** No usable follow-up pairs exist in the dataset, so response calls are simulated from a bias model fitted to 101 real per-lesion measurements (36 in bright-tumour cases): the median error of detected lesions is -10.9 mm at the default threshold and +0.6 mm recalibrated, and detection falls with size and for bright tumours. The simulation needs one assumption it cannot fit: how much a lesion's detection draw carries over between the two visits (persistence; 0 = independent misses, 1 = the same every visit). Across 304 simulated scenarios the share in which the default and recalibrated thresholds give different RECIST categories in at least 25% of draws is 53% at persistence 0, 49% at 0.5 and 43% at 1; the default is wrong more often than the recalibrated measurement in 94% of scenarios at every setting. For an individual scenario the disagreement probability moves by about 10 percentage points on average between persistence 0 and 1. So the qualitative finding (in a substantial share of simulated scenarios the response call depends on the calibration) does not hinge on the assumption, while any single scenario's probabilities do. The fitted measurement noise is large (SD 16 to 19 mm), so even the recalibrated measurement often miscategorises small true changes. These figures describe what measurement error could do to a response call, not any patient.

## 8. Are the findings specific to VISTA-3D? Two more segmenters (`results/w3_models/`; rules in `PREREG_stage1.md` and `PREREG_stage2.md`, deviations in `DEVIATIONS.md`; HCC cohort 74 cases / 85 lesions)
- Behaviour (Stage 1, [STAGE1_SUMMARY.md](w3_models/stage1/STAGE1_SUMMARY.md)): the Swin UNETR (A) and a self-trained nnU-Net (D) both under-detect bright tumours (detection gaps +59.2 and +64.5 points).
- Mechanism and causality (Stage 2, [STAGE2_SUMMARY.md](w3_models/stage2/STAGE2_SUMMARY.md)): both models are LOST LATE (represented in the encoder, not decoded by the head); erasing the tumour direction costs both about the same lesion Dice (PARTIAL REPLICATION: A is flagged on the probe check); editing lesion brightness changes detection in the hypothesised direction in both arms in both models (M3 SUPPORTED); a single threshold shift does not generalise across both models under the pre-registered grid (both PARTIAL), and an exploratory wider grid helps A more than D.
- Cannot say: that the training data cause the gap (W5, pre-registered, not yet run); anything beyond this cohort; anything about patients.

## What can and cannot be said
- Can: the model under-measures hepatic tumour diameters by about a fifth; the missed tissue is represented in the encoder and lost downstream; the model's detection depends on contrast sign (causally, by editing); a single threshold shift removes most of the bias on this cohort, and our 48-weight re-fit did not beat it (this does not exclude a better readout).
- Cannot: anything about patients or outcomes; anything about phase; that the threshold shift generalises beyond HCC-TACE-Seg (it was chosen on HCC cases only); that the bias is the same on other cohorts (the research in this file is HCC-TACE-Seg only).
- Pre-registration record: PREREG.md / DEVIATIONS.md in each results folder, committed before the corresponding results.
