# W6 / W4b results (living file; research use only, no clinical claims)

Last updated 2026-11-09 06:50 EST. Frozen analysis (`code_hashes_analysis.txt`); every number below comes from files in this folder. Single training runs per arm (seed 2 pending). Models are internal derivatives of NV-Segment-CT and are never published.

## Arm status

| Arm | Trained | WAW scored | HCC | Forgetting (organs, side check, false positives) | Drift check |
|---|---|---|---|---|---|
| BRIGHT (full fine-tune) | yes | yes | yes | yes | not run |
| SHAM (darkened) | yes | yes | yes | yes | not run |
| PLAIN (no edit) | yes | yes | yes | kidney Dice only (rest pending) | running |
| BRIGHT-DEC (decoder only) | yes | yes | yes | kidney Dice only (rest pending) | running |
| BRIGHT seed 2 | training (started 06:33 EST) | - | - | - | - |
| TotalSegmentator | running (started 06:34 EST) | - | - | not applicable | - |

The drift check is blocked: the frozen script cannot hold the drift masks of 20 series in memory (DEVIATIONS 31). A streaming candidate gives a byte-identical output on a real 2-series test and is unused until the researcher signs it off.

## 1. W4b primary (pre-registered): BRIGHT-FT minus SHAM-FT

theta difference d = **-0.277 [-0.418, -0.135]: REDUCES THE PENALTY** (297 tumours; F1 81, NF 211; one-sided alpha 0.025). Full report: `W4B_PRIMARY_REPORT.md`.

## 2. W4b secondaries and per-model labels

| Model | theta [95% CI] | Label | NO PENALTY test (margin 0.25) |
|---|---|---|---|
| BASELINE (already seen) | +0.159 [+0.036, +0.280] | INCONCLUSIVE | INCONCLUSIVE |
| A (already seen) | +0.314 [+0.188, +0.439] | BRIGHT PENALTY | INCONCLUSIVE |
| D0 (already seen) | +0.352 [+0.222, +0.483] | BRIGHT PENALTY | INCONCLUSIVE |
| B50 (already seen) | -0.060 [-0.184, +0.056] | INCONCLUSIVE | NO PENALTY |
| BRIGHT | -0.032 [-0.164, +0.100] | INCONCLUSIVE | NO PENALTY |
| SHAM | +0.245 [+0.121, +0.379] | BRIGHT PENALTY | INCONCLUSIVE |
| PLAIN | +0.209 [+0.089, +0.338] | BRIGHT PENALTY | INCONCLUSIVE |

Secondary S1, BRIGHT minus PLAIN: **d = -0.241 [-0.384, -0.090], REDUCES THE PENALTY.** Fine-tuning without the edit (PLAIN) keeps the penalty, so the reduction is not just an effect of fine-tuning on this task.
S2 (BRIGHT-DEC minus SHAM): not run yet (arm unscored in the analysis). S3 (BRIGHT NO PENALTY): NO PENALTY under the primary contrast definition.

Sensitivities that change a verdict: with the liver eroded by 3 mm, BRIGHT's NO PENALTY becomes INCONCLUSIVE, and PLAIN's BRIGHT PENALTY becomes INCONCLUSIVE. All other sensitivities change nothing. So "BRIGHT has no penalty" is not robust to that sensitivity.

## 3. Bright-tumour detection on the arterial series (W6 arterial test)

297 tumours; bright = arterial contrast above 0 (242 tumours); dark = 55.

| Model | Bright tumours detected |
|---|---|
| BASELINE | 21.5% (52 of 242) |
| SHAM | 21.5% (52 of 242) |
| PLAIN | 14.0% (34 of 242) |
| **BRIGHT** | **77.3% (187 of 242)** |

Baseline and SHAM happen to have the same count; their dark-minus-bright gaps differ (0.35 vs 0.59).
Paired tests (the W5 CLOSES rule): BRIGHT vs SHAM CONFIRMED, detection delta -0.612 [-0.708, -0.530], Dice delta -0.271 [-0.325, -0.216]. BRIGHT vs PLAIN CONFIRMED, detection delta -0.541 [-0.637, -0.432], Dice delta -0.233 [-0.292, -0.173]. BRIGHT vs BASELINE (descriptive, confounded with fine-tuning) CONFIRMED.

## 4. Forgetting

Guard rules (PREREG): organ Dice drop at most 0.02 (mean) with the CI bound above -0.03; drift Dice at least 0.95 only above 10 mL; false positives median at most 1.25 times baseline plus 0.5 mL.

| Measure (baseline) | BRIGHT | SHAM | PLAIN | BRIGHT-DEC |
|---|---|---|---|---|
| Spleen Dice (0.9624, n 41) | 0.9612 pass | 0.9505 pass | pending | pending |
| Kidney Dice (0.8742, n 60) | 0.8182 **fail** (-0.056 [-0.065, -0.048]) | 0.8297 **fail** (-0.045 [-0.053, -0.037]) | 0.842 (-0.032, provisional) | 0.844 (-0.030, provisional) |
| Liver Dice (0.9692, n 27) | 0.9598 pass | 0.9622 pass | pending | pending |
| Kidney-tumour side check (0.5165, 65 lesions, no criterion) | 0.4941 | 0.5006 | pending | pending |
| False positives, median mL (baseline 0.05) | 0.55 pass | 0.18 pass | pending | pending |
| Drift (20 series) | not run | not run | not run | not run |

BRIGHT and SHAM both carry the tag **FORGETTING COST: organ_kidneys**. The unedited PLAIN arm also loses kidney Dice, so the loss comes from tumour-only fine-tuning with the kidney class unsupervised and not from the brightening edit.
Files: `guard/forget_analysis_{BRIGHT,SHAM}_nodrift.json`.

## 5. Limits

Single runs per arm (BRIGHT seed 2 pending); drift not yet checked; WAW-TACE is the validation set for the primary, and the HCC-TACE-Seg results are computed on the instance but not yet summarised here; PLAIN and BRIGHT-DEC forgetting are provisional until the frozen analysis runs; the second-half W4 models labelled "already seen" were analysed before W4b existed.

## 6. Deviations from tonight

DEVIATIONS 26 to 32 in `../w3_models/DEVIATIONS.md` (false stall hold, logging-only fix, TotalSegmentator test, drift class names, HCC analysis folder, drift memory problem, shared-GPU scheduling).
