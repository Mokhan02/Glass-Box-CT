# W6 / W4b results (living file; research use only, no clinical claims)

Last updated 2026-11-09 15:10 EST. Frozen analysis (`code_hashes_analysis.txt`); every number below comes from files in this folder. Single training runs per arm (seed 2 pending). Models are internal derivatives of NV-Segment-CT and are never published.

## Arm status

| Arm | Trained | WAW scored | HCC | Forgetting (organs, side check, false positives) | Drift check |
|---|---|---|---|---|---|
| BRIGHT (full fine-tune) | yes | yes | yes | yes | done: FAILS (32 of 48 classes) |
| SHAM (darkened) | yes | yes | yes | yes | done: FAILS (29 of 48) |
| PLAIN (no edit) | yes | yes | yes | yes | done: FAILS (29 of 48) |
| BRIGHT-DEC (decoder only) | yes | yes | yes | yes | done: FAILS (20 of 48 classes) |
| BRIGHT seed 2 | yes (08:07 EST) | yes (10:47 EST) | yes | yes | done: FAILS (31 of 48 classes) |
| TotalSegmentator | running (176 of 200 series of half 0 at 10:45 EST; ~95 s per series) | - | - | not applicable | - |

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
| BRIGHT seed 2 | -0.059 [-0.191, +0.090] | INCONCLUSIVE | NO PENALTY |

Secondary S1, BRIGHT minus PLAIN: **d = -0.241 [-0.384, -0.090], REDUCES THE PENALTY.** Fine-tuning without the edit (PLAIN) keeps the penalty, so the reduction is not just an effect of fine-tuning on this task.
Secondary S2, BRIGHT-DEC minus SHAM: **d = -0.233 [-0.375, -0.079], REDUCES THE PENALTY.**
Decoder-only ratio R (share of BRIGHT's gap-closing that the decoder-only arm achieves): detection 0.82 [0.70, 0.95], Dice 0.88 [0.77, 1.00]; both DECODER SUFFICES. S3 (BRIGHT NO PENALTY): NO PENALTY under the primary contrast definition.

**Replication (BRIGHT seed 2 vs SHAM): CONFIRMED (detection delta -0.640 [-0.745, -0.539], Dice -0.269 [-0.321, -0.216]).** The two BRIGHT seeds do not differ in their gap closing (detection difference of gaps +0.028 [-0.021, +0.084]; Dice -0.002 [-0.015, +0.012]).
Files: `analysis_all_arms/` (all arms including seed 2) and `analysis_with_plain_dec/` (primary.json, sensitivity_table.json, ANALYSIS.md, detection_table.json). Sensitivities that change a verdict (checked with PLAIN; rerun with DEC to confirm): with the liver eroded by 3 mm, BRIGHT's NO PENALTY becomes INCONCLUSIVE, and PLAIN's BRIGHT PENALTY becomes INCONCLUSIVE. All other sensitivities change nothing. So "BRIGHT has no penalty" is not robust to that sensitivity.

## 3. Detection by contrast group (297 tumours; bright = contrast on that series above 0; detection = W4 `detected`)

| Arm | Arterial dark (n 55) | Arterial bright (n 242) | Portal-venous bright (n 166) | Portal-venous dark (n 131) |
|---|---|---|---|---|
| BASELINE | 56.4% (31) | 21.5% (52) | 21.1% (35) | 69.5% (91) |
| SHAM | 80.0% (44) | 21.5% (52) | 27.1% (45) | 78.6% (103) |
| PLAIN | 65.5% (36) | 14.0% (34) | 18.7% (31) | 67.9% (89) |
| **BRIGHT** | 74.5% (41) | **77.3% (187)** | 65.7% (109) | 78.6% (103) |
| BRIGHT seed 2 | 70.9% (39) | 76.4% (185) | 60.2% (100) | 78.6% (103) |
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
| BRIGHT seed 2 | 90.6% / 0.607 | 77.8% / 0.431 | 96.6% / 0.689 |
| BRIGHT-DEC | 91.8% / 0.604 | 77.8% / 0.401 | 98.3% / 0.698 |

The baseline reference row has 87 lesions and the arms 86 (85 common); one baseline-only lesion was dropped. No confidence intervals here (descriptive; the per-arm gap analyses are in each arm's `hcc/analysis`).

## 3c. Over-segmentation check (descriptive; `overseg/`, 396 WAW-TACE series, 394 with a ground-truth tumour)

| Arm | False-positive tumour volume strictly outside the GT: median / mean mL per series | Predicted / GT tumour volume: median ratio | Median predicted tumour volume (mL) |
|---|---|---|---|
| BASELINE | 0.51 / 3.49 | 0.16 | 6.1 |
| SHAM | 0.74 / 5.51 | 0.23 | 9.9 |
| PLAIN | 0.28 / 2.40 | 0.11 | 5.0 |
| **BRIGHT** | **2.56 / 10.79** | **0.68** | 28.0 |
| BRIGHT-DEC | 1.94 / 8.07 | 0.57 | 21.0 |

BRIGHT predicts about four times the baseline's tumour volume per series and about five times its false-positive volume outside the ground truth (median 2.56 vs 0.51 mL). Its median predicted/GT ratio is 0.68, still below 1, so across series it is not over-segmenting in total volume; the baseline and SHAM under-segment strongly (0.16, 0.23). The guard's 10 mm false-positive measure (median 0.05 baseline, 0.55 BRIGHT, 0.18 SHAM) passes for both. The extra volume is partly the tumour boundary (the ground truth is a drawn mask) and partly real extra prediction; this table cannot separate the two.

## 3d. Over-segmentation detail (descriptive; `overseg2/`, 396 series; `scripts/w6_overseg2.py`)

Predicted connected components (26-connectivity) with no overlap with any GT tumour voxel and a size of at least 0.5 mL:

| Arm | Per series: median / mean | % of series with at least one |
|---|---|---|
| BASELINE | 0 / 0.38 | 28.0% |
| SHAM | 0 / 0.50 | 31.3% |
| PLAIN | 0 / 0.39 | 27.3% |
| **BRIGHT** | 0 / 0.70 | **39.6%** |
| BRIGHT-DEC | 0 / 0.71 | 40.9% |

False-positive volume (predicted voxels outside every GT tumour voxel) by distance to the nearest GT tumour voxel, median / mean mL per series:

| Arm | 0-5 mm | 5-10 mm | >10 mm |
|---|---|---|---|
| BASELINE | 0.00 / 1.40 | 0.00 / 0.38 | 0.05 / 1.71 |
| SHAM | 0.02 / 1.70 | 0.00 / 0.56 | 0.18 / 3.25 |
| PLAIN | 0.00 / 0.81 | 0.00 / 0.22 | 0.10 / 1.36 |
| **BRIGHT** | 0.63 / 3.26 | 0.00 / 1.09 | 0.55 / 6.43 |
| BRIGHT-DEC | 0.42 / 3.01 | 0.00 / 0.96 | 0.56 / 4.10 |

Reading: BRIGHT's extra false-positive volume sits both next to the tumours (0-5 mm: median 0.63 vs 0.00 mL at baseline) and far from them (>10 mm: median 0.55 vs 0.05, mean 6.43 vs 1.71). About 40% of series carry at least one separate predicted component of 0.5 mL or more with no overlap with the GT, against 28% at baseline. The >10 mm band reproduces the guard's false-positive medians (0.05, 0.18, 0.55). Some of these components may be real tumour tissue that the drawn ground truth does not cover; this table cannot tell.

## 3e. EXPLORATORY: detection at matched false-alarm rate (descriptive; `matched_fa/`; `scripts/w6_matched_fa.py`)

The W4-route predictions keep only masks (no probability maps), so a minimum component size stands in for a threshold. Reference: the baseline has a no-overlap predicted component of at least 0.5 mL in 111 of 396 series (28.0%). For each arm the cut s is the smallest size at which at most 111 series keep a no-overlap component larger than s; all components of size at most s are removed (overlapping or not, blind to the ground truth), and the predictions are rescored with the unchanged W4 scorer. Cuts: BRIGHT 1.43 mL, BRIGHT-DEC 1.21 mL, SHAM 0.68 mL (unfiltered rates 157, 162 and 124 series; PLAIN is already at 108).

| Arm (matched) | Arterial dark (n 55) | Arterial bright (n 242) | Portal bright (n 166) | Portal dark (n 131) |
|---|---|---|---|---|
| BASELINE (unfiltered) | 56.4% (31) | 21.5% (52) | 21.1% (35) | 69.5% (91) |
| SHAM, s 0.68 mL | 76.4% (42) | 20.7% (50) | 26.5% (44) | 72.5% (95) |
| BRIGHT, s 1.43 mL | 70.9% (39) | 68.2% (165) | 56.6% (94) | 71.8% (94) |
| BRIGHT-DEC, s 1.21 mL | 69.1% (38) | 57.9% (140) | 47.0% (78) | 71.8% (94) |

At the baseline's false-alarm rate BRIGHT still detects 68.2% of arterial-bright tumours (77.3% unfiltered) against 21.5% for the baseline, and DEC 57.9% (67.8% unfiltered). The size filter costs BRIGHT about 9 points and DEC about 10 points of bright detection. This is a size-cut proxy, not a threshold sweep, it is not pre-registered, and no uncertainty intervals are given.

## 4. Forgetting

Guard rules (PREREG): organ Dice drop at most 0.02 (mean) with the CI bound above -0.03; drift Dice at least 0.95 for every class above the 10 mL floor; false positives median at most 1.25 times baseline plus 0.5 mL. Streaming drift script (DEVIATIONS 33); its non-drift parts are identical to the earlier no-drift runs.

| Measure (baseline) | BRIGHT | SHAM | PLAIN | BRIGHT-DEC |
|---|---|---|---|---|
| Spleen Dice (0.9624, n 41) | 0.9612 pass | 0.9505 pass | 0.9466 pass | 0.9623 pass |
| Kidney Dice (0.8742, n 60) | 0.8182 **fail** (-0.056 [-0.065, -0.048]) | 0.8297 **fail** (-0.045 [-0.053, -0.037]) | 0.8420 **fail** (-0.032) | 0.8444 **fail** (-0.030 [-0.033, -0.027]) |
| Liver Dice (0.9692, n 27) | 0.9598 pass | 0.9622 pass | 0.9628 pass | 0.9639 pass |
| Kidney-tumour side check (0.5165, 65 lesions, no criterion) | 0.4941 | 0.5006 | 0.4879 | 0.5165 (identical to baseline, see note) |
| False positives, 10 mm, median mL (baseline 0.05) | 0.55 pass | 0.18 pass | 0.10 pass | 0.56 pass |
| Drift: classes above 10 mL failing 0.95 (of 48) | **32** fail | **29** fail | **29** fail | **20** fail |
| Drift: lowest class mean Dice above the floor | 0.748 | 0.811 | 0.783 | 0.785 |
| Guard result | FORGETTING COST: drift, organ_kidneys | FORGETTING COST: drift, organ_kidneys | FORGETTING COST: drift, organ_kidneys | FORGETTING COST: drift, organ_kidneys |

All three fine-tuned arms, including the unedited PLAIN arm, fail the drift criterion on 29 to 32 of the 48 classes above the floor and lose kidney Dice. The failure is therefore a property of the tumour-only fine-tuning recipe, not of the brightening edit. BRIGHT seed 2 behaves like seed 1 on the guard: spleen 0.9547 pass, kidney 0.8246 fail (-0.050 [-0.057, -0.043]), liver 0.9644 pass, false positives median 0.42 mL pass, side check 0.4950, 31 of 48 classes failing the drift criterion (lowest class mean Dice 0.749): FORGETTING COST: drift, organ_kidneys. BRIGHT is slightly worse than SHAM and PLAIN on drift (32 vs 29 classes); the decoder-only arm drifts least (20 classes) and loses the least kidney Dice (-0.030). DEC's kidney-tumour side-check file is byte-identical to the baseline's: the point route used for that check apparently does not use the parameters the decoder-only arm trained (checked in the code: the point path calls the shared encoder's `up_layers` and `point_head`, while the decoder-only arm trains only `image_encoder.up_layers_auto` and `class_head`, so it cannot change the point route; MONAI `vista3d.py`, `segresnet_ds.py`). No arm passes the guard, so every result is tagged "fix with forgetting cost". Files: `guard/stream/`, `guard/forget_analysis_*_nodrift.json`.

## 5. Limits

Single runs per arm (BRIGHT seed 2 pending); drift not yet checked; WAW-TACE is the validation set for the primary, and the HCC-TACE-Seg results are computed on the instance but not yet summarised here;  the HCC table has no uncertainty intervals; the second-half W4 models labelled "already seen" were analysed before W4b existed.

## 6. Deviations from tonight

DEVIATIONS 26 to 35 in `../w3_models/DEVIATIONS.md` (false stall hold, logging-only fix, TotalSegmentator test, drift class names, HCC analysis folder, drift memory problem, shared-GPU scheduling, streaming drift script).
