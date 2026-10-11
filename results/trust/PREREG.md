# Pre-registration: contrast-phase check and flag-gated recalibration

Written and committed before any phase estimate was computed on HCC-TACE-Seg and before the gating analysis was run. Research use only; heuristics, not diagnoses. Product features that use these results are labelled "research heuristic" in the app. Deviations go in `results/trust/DEVIATIONS.md` with their timing.

## A. Contrast-phase estimate (heuristic)

**Inputs.** The model's own segmentation (no reference masks): class 6 (aorta) and class 17 (portal vein and splenic vein) of NV-Segment-CT, prompted explicitly. Raw HU of the CT. Each mask is eroded by one voxel (1.5 mm) to avoid partial-volume edges; the aorta is restricted to the slab from 20 mm below to 20 mm above the predicted liver (z-extent of class 1 at p > 0.5) so that arch and iliac segments do not enter. A vessel is "not measurable" if fewer than 200 voxels (at 1.5 mm) remain. The statistic is the **median HU** inside the mask.

**Thresholds, fixed now from the literature** (and not from the HCC data):
- Late-arterial aortic attenuation targets are about 250 to 300 HU in the RSNA liver CT protocol audit, and arterial-phase aortic opacification of about 245 to 254 HU is measured in a clinical series (Journal of Clinical Imaging Science, 2023, on reduced contrast dose). Unenhanced blood is about 35 to 50 HU, so an aorta below 100 HU is essentially unenhanced. The enhancement-based rule below uses a deliberately lower arterial cut (200 HU) than the 250 to 300 HU targets to tolerate lower peaks; no cut was tuned on HCC-TACE-Seg.
- No published HU rule for four-way phase was found; automatic four-phase classifiers exist but are machine-learned (aorta/portal vein/IVC radiomics, European Radiology 2024). This rule is therefore a **coarse research heuristic**, not a validated classifier.

**Rule (in this order).**
1. aorta not measurable -> `indeterminate`.
2. aorta median < 100 HU -> `unenhanced`.
3. aorta median >= 200 HU -> `arterial-like`.
4. 100 <= aorta < 200 and portal vein median >= 100 HU -> `venous-like` (portal-venous or later).
5. otherwise -> `indeterminate`.
DICOM tags (contrast agent, bolus start time, acquisition time, series description) are read when present and reported next to the estimate; a tag-versus-HU disagreement raises "check the phase" and never overrides the HU rule. HCC-TACE-Seg has these fields stripped (earlier audit: contrast agent present in 1 of 89 cases), so they are not evaluated here.

**Blind-spot warning.** Fires when the phase estimate is `arterial-like` OR the contrast measured on the predicted tumour (tumour minus liver more than 5 mm away) is greater than 0 HU. It means "this looks like the situation in which this model tends to miss tumours" and nothing more.

**Evaluation (descriptive only; no threshold was tuned and nothing is gated on it).** Clean 89-case cohort, the model's own aorta and portal-vein masks. Reported with patient-bootstrap 95% CIs (2000 resamples, seed 1):
1. the distribution of phase estimates and how many vessels were measurable;
2. agreement between `arterial-like` and the sign of the **reference** tumour-minus-liver contrast (reference > 0 HU), as a 2 x 2 table, percent agreement and Cohen's kappa; likewise `venous-like` against reference <= 0;
3. the **miss rate** (largest reference lesion covered < 10% by the default prediction, `results/readout_refit/bright_threshold_lesions.csv`) when the blind-spot warning fires versus when it does not, and when the phase estimate alone is arterial-like versus not.
Limits stated in advance: one disease, one dataset; phase labels do not exist, so agreement is with a physical proxy (contrast sign), not with true phase; arterial-like cases may be few.

## B. Flag-gated recalibration

**Question.** Is applying the recalibrated threshold only when the quality flag predicts under-segmentation better than always using the default or always recalibrating?

**Data.** Saved class-26 logits of the clean 89-case cohort (`cache/rr`, preprocessed 1.5 mm grid, annotated slices), 5-fold patient-level CV (cases sorted by id, fold = index % 5, the same folds as `results/readout_refit`).

**Policies.** D: predicted tumour = logit > 0 (default). R: logit > -3.0 (the recalibrated threshold the app uses). G (gated): R if the flag predicts under-segmentation, else D; an empty default prediction counts as "predicts under-segmentation".

**Flag inside CV.** Features of the default prediction exactly as the shipped flag (`log1p(volume cm3)`, mean probability, mean binary entropy over predicted-tumour voxels, voxel volume 3.375 mm3); logistic regression on standardised features (C = 1) with label "tumour Dice at the default threshold < 0.7", fitted on **training folds only**; the flag threshold is the largest probability whose sensitivity for the label on the training folds is >= 90% (the shipped rule, computed in-sample on training folds because the cohort is small). The shipped flag model is not used here because it was fitted on the same cases.

**Outcomes (held-out folds, per case).** The relative error of the sum of target diameters (the two largest reference lesions >= 10 mm; lesion diameters and detection as in `results/readout_refit`), its absolute value, the indicator "over-estimated by >= 20%", the indicator "|error| >= 20%", and tumour Dice. Cases with no measurable target are excluded for the diameter outcomes.

**Decision rule (fixed now).** The gated policy is **shipped** only if BOTH hold, using patient-bootstrap (2000 resamples, seed 1) 95% CIs of paired per-case differences:
1. over-estimation: the mean of (indicator_G - indicator_R) has a CI entirely below 0 (fewer >= 20% over-estimates than always-recalibrated);
2. accuracy: the mean of (|error_G| - |error_D|) has a CI entirely below 0 (smaller absolute sum-of-diameters error than always-default).
Otherwise: **not shipped**; the app keeps the side-by-side default / recalibrated view. All three policies are reported either way, with the median signed error (bias), the >= 20% over-estimation rate, the >= 20% absolute error rate and Dice.

**Stated limitations.** The -3.0 threshold was chosen from earlier cross-validated work on these same cases, so R and G are mildly optimistic relative to a fresh cohort; the flag threshold is chosen in-sample on training folds; about 89 cases give wide intervals; saved logits are on the preprocessed grid on annotated slices, not in the original geometry.
