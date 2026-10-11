# Experiment 8: what distinguishes missed from detected hepatic-tumour lesions?

Data: 115 RECIST target lesions (>= 10 mm; 2 largest per case) in 103 HCC-TACE-Seg cases, all `in_distribution = False`. Detected = predicted voxels cover >= 10% of the lesion (86 detected, 29 missed).
Covariates computed by `scripts/exp8_lesion_covariates.py` (table: `lesion_table.csv`), models by `scripts/exp8_fit.py` (`models.csv`, `strata_30_60mm.csv`). The design was fixed before the table was read.
Detection counts reproduce the RECIST run exactly (86 of 115), so the lesion definition matches.

**Data-quality caveat that applies throughout:** 28 cases are geometry-route (label alignment unverified; see `results/exp1_audit/AUDIT.md` section 5), 22 series mix acquisitions, and 3 single-acquisition geometry cases (HCC_006, 015, 016)
were converted from the wrong series. The **UID-route subset (75 cases, 86 lesions, 16 missed) is the cleanest** and is reported alongside the pooled and single-acquisition views. The corrected conversion is being evaluated separately.

## 1. Pre-specified: does anything beat size alone? No.
Cross-validated AUROC (5-fold grouped by case, repeated 20x; case-bootstrap 95% CI); gain over size alone in brackets.

| model | all lesions (115; 29 missed) | UID route (86; 16 missed) | single-acquisition (92; 21 missed) |
|---|---|---|---|
| size only (log diameter) | 0.741 [0.62, 0.87] | 0.673 [0.49, 0.84] | 0.701 [0.55, 0.84] |
| + contrast (absolute tumour-liver HU) | -0.024 [-0.045, -0.009] | -0.016 [-0.077, +0.049] | -0.021 [-0.045, -0.004] |
| + phase (aorta HU, whole-liver HU) | -0.008 [-0.081, +0.057] | -0.011 [-0.164, +0.117] | +0.016 [-0.104, +0.118] |
| + window position (edge distance, covering windows, contained) | -0.014 [-0.065, +0.026] | -0.050 [-0.134, +0.011] | -0.054 [-0.113, -0.013] |
| + window content (liver and tumour volume in window) | -0.022 [-0.067, +0.012] | -0.053 [-0.105, -0.011] | -0.025 [-0.084, +0.031] |
| all blocks (exploratory) | -0.057 [-0.144, +0.023] | -0.146 [-0.310, +0.000] | -0.084 [-0.212, +0.025] |

Median diameter, missed vs detected: 40 vs 80 mm. **Power is limited**: with 16-29 missed lesions only large effects are detectable (a planted effect of moderate size in a synthetic test of the same script gave a gain whose interval still included 0 at 140 lesions).
A negative here means "no large effect detectable", not "no effect".

## 2. The specific prediction (30-60 mm lesions: missed ones have low contrast or the wrong phase)
UID route, 8 missed vs 22 detected (all lesions: 17 vs 26). Mann-Whitney, unadjusted; 9 variables x 3 subsets = 27 tests, Holm threshold for 9 variables about 0.006, so **nothing below survives correction**.

| variable | missed median | detected median | p |
|---|---|---|---|
| \|contrast\| (HU) | 26.9 | 13.5 | 0.013 (direction opposite to the prediction) |
| share with \|contrast\| < 20 HU (LiTS cut, not tuned) | 12% | 73% | - |
| aorta HU (arterial proxy) | 205 | 160 | 0.47 |
| liver HU near lesion | 95 | 108 | 0.11 |
| best distance to window edge (mm) | 48.5 | 54.1 | 0.50 |
| covering windows | 4 | 4 | 0.78 |
| liver volume in the window (cm3) | 1341 | 1453 | 0.98 |

- **Low contrast does not explain the misses; the data point the other way.**
- **Window position and window content (the architecture-level explanations, including instance normalisation) show no association.**
- Phase is suggestive only (arterial proxy higher and surrounding liver darker in missed lesions; not significant).

## 3. Exploratory, NOT pre-specified: the SIGN of contrast (`exploratory_signed_contrast.txt`)
Added after seeing that magnitude pointed the wrong way. Detection rate by whether the tumour is brighter or darker than the surrounding liver:

| | 30-60 mm: brighter | 30-60 mm: darker | >= 60 mm: brighter | >= 60 mm: darker |
|---|---|---|---|---|
| all lesions | 7/20 (0.35) | 19/23 (0.83) | 10/15 (0.67) | 48/49 (0.98) |
| UID route | 7/13 (0.54) | 15/17 (0.88) | 7/11 (0.64) | 39/39 (1.00) |

Unadjusted Fisher p: 0.0020, 0.0020 (all lesions), 0.049, 0.0014 (UID). The >= 60 mm stratum was not used to form the idea, so it is mildly supportive, but it is the same cohort, so it is not independent confirmation.
Against this: adding a *linear* signed-contrast term to size gives no cross-validated gain (+0.002 [-0.08, +0.08] all; -0.002 [-0.18, +0.12] UID). A binary bright/dark indicator was not modelled, to avoid more unplanned looks.
**Hypothesis to test on new data:** hyperattenuating (brighter-than-liver) HCC is missed far more often than hypoattenuating HCC, plausibly because tumour training examples are mostly hypodense (arterial-phase hyperenhancement would be a distribution shift).
This needs a pre-registered test on data not used here; nothing in this repository supports it beyond the numbers above.

## Not tested
Necrotic fraction, tiling shift reruns and per-lesion residual search (the rest of the strategy document's Experiment 8): not run.
