# W6 / W4b results (living file; research use only, no clinical claims)

Last updated 2026-11-09 07:00 EST. Frozen analysis (`code_hashes_analysis.txt`); every number below comes from files in this folder. Single training runs per arm (seed 2 pending). Models are internal derivatives of NV-Segment-CT and are never published.

## Arm status

| Arm | Trained | WAW scored | HCC | Forgetting (organs, side check, false positives) | Drift check |
|---|---|---|---|---|---|
| BRIGHT (full fine-tune) | yes | yes | yes | yes | streaming run in progress |
| SHAM (darkened) | yes | yes | yes | yes | streaming run in progress |
| PLAIN (no edit) | yes | yes | yes | done on the instance, analysis pending | pending |
| BRIGHT-DEC (decoder only) | yes | yes | yes | drift inference running (11 of 20 series) | pending |
| BRIGHT seed 2 | training (epoch 10 of 77 at 06:50 EST) | - | - | - | - |
| TotalSegmentator | running (10 of 200 series of half 0 at 06:50 EST) | - | - | not applicable | - |

The frozen forgetting script cannot hold the drift masks of 20 series in memory (DEVIATIONS 31). The researcher approved a streaming version (memory only; byte-identical output on a real 2-series test; DEVIATIONS 33), now running for the drift check.

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
| BRIGHT-DEC | +0.013 [-0.127, +0.152] | INCONCLUSIVE | NO PENALTY |

Secondary S1, BRIGHT minus PLAIN: **d = -0.241 [-0.384, -0.090], REDUCES THE PENALTY.** Fine-tuning without the edit (PLAIN) keeps the penalty, so the reduction is not just an effect of fine-tuning on this task.
Secondary S2, BRIGHT-DEC minus SHAM: **d = -0.233 [-0.375, -0.079], REDUCES THE PENALTY.**
Decoder-only ratio R (share of BRIGHT's gap-closing that the decoder-only arm achieves): detection 0.82 [0.70, 0.95], Dice 0.88 [0.77, 1.00]; both DECODER SUFFICES. S3 (BRIGHT NO PENALTY): NO PENALTY under the primary contrast definition.

Files: `analysis_with_plain_dec/` (primary.json, sensitivity_table.json, ANALYSIS.md, detection_table.json). Sensitivities that change a verdict (checked with PLAIN; rerun with DEC to confirm): with the liver eroded by 3 mm, BRIGHT's NO PENALTY becomes INCONCLUSIVE, and PLAIN's BRIGHT PENALTY becomes INCONCLUSIVE. All other sensitivities change nothing. So "BRIGHT has no penalty" is not robust to that sensitivity.

## 3. Detection by contrast group (297 tumours; bright = contrast on that series above 0; detection = W4 `detected`)

| Arm | Arterial dark (n 55) | Arterial bright (n 242) | Portal-venous bright (n 166) | Portal-venous dark (n 131) |
|---|---|---|---|---|
| BASELINE | 56.4% (31) | 21.5% (52) | 21.1% (35) | 69.5% (91) |
| SHAM | 80.0% (44) | 21.5% (52) | 27.1% (45) | 78.6% (103) |
| PLAIN | 65.5% (36) | 14.0% (34) | 18.7% (31) | 67.9% (89) |
| **BRIGHT** | 74.5% (41) | **77.3% (187)** | 65.7% (109) | 78.6% (103) |
| BRIGHT-DEC | 74.5% (41) | 67.8% (164) | 55.4% (92) | 78.6% (103) |

BRIGHT detects fewer dark arterial tumours than SHAM (41 vs 44 of 55); its gain on bright tumours carries over to the portal-venous series (65.7% vs 21.1% at baseline). Baseline and SHAM happen to have the same arterial-bright count; their dark-minus-bright gaps differ (0.35 vs 0.59).
Paired tests (W5 CLOSES rule): BRIGHT vs SHAM CONFIRMED (detection delta -0.612 [-0.708, -0.530], Dice -0.271 [-0.325, -0.216]); BRIGHT vs PLAIN CONFIRMED (-0.541 [-0.637, -0.432]; Dice -0.233 [-0.292, -0.173]); DEC vs SHAM CONFIRMED (-0.517 [-0.612, -0.436]; Dice -0.243 [-0.291, -0.192]); BRIGHT vs BASELINE CONFIRMED (descriptive, confounded with fine-tuning).

## 3b. HCC-TACE-Seg (85 lesions matched across all five arms; detection = coverage at least 0.10; Dice = full-volume lesion Dice)

| Arm | All: detected / Dice | Bright (n 27) | Dark (n 58) |
|---|---|---|---|
| BASELINE | 81.2% / 0.525 | 51.9% / 0.251 | 94.8% / 0.653 |
| SHAM | 82.4% / 0.535 | 51.9% / 0.248 | 96.6% / 0.669 |
| PLAIN | 76.5% / 0.478 | 37.0% / 0.178 | 94.8% / 0.618 |
| **BRIGHT** | 94.1% / 0.622 | 85.2% / 0.447 | 98.3% / 0.704 |
| BRIGHT-DEC | 91.8% / 0.604 | 77.8% / 0.401 | 98.3% / 0.698 |

The baseline reference row has 87 lesions and the arms 86 (85 common); one baseline-only lesion was dropped. No confidence intervals here (descriptive; the per-arm gap analyses are in each arm's `hcc/analysis`).

## 3c. Over-segmentation check (descriptive)

Pending: `scripts/w6_overseg.py` (false-positive tumour volume strictly outside the ground-truth tumours, median mL per series; predicted/GT tumour volume ratio) is running. The 10 mm guard measure so far: median 0.05 mL (baseline), 0.55 (BRIGHT), 0.18 (SHAM).

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

Single runs per arm (BRIGHT seed 2 pending); drift not yet checked; WAW-TACE is the validation set for the primary, and the HCC-TACE-Seg results are computed on the instance but not yet summarised here; PLAIN and BRIGHT-DEC forgetting are provisional until the frozen analysis runs; the HCC table has no uncertainty intervals; the second-half W4 models labelled "already seen" were analysed before W4b existed.

## 6. Deviations from tonight

DEVIATIONS 26 to 33 in `../w3_models/DEVIATIONS.md` (false stall hold, logging-only fix, TotalSegmentator test, drift class names, HCC analysis folder, drift memory problem, shared-GPU scheduling, streaming drift script).
