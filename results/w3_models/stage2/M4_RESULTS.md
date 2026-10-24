# W3 Stage 2, M4: does a nested-CV threshold shift fix A and D as it fixed VISTA-3D? (PREREG_stage2.md, approved with three edits)

Method (as pre-registered): the tumour decision margin saved in the M2 baseline pass (A: class-27 logit; D: tumour logit minus the max of the other logits; 0 = the model's default), on each model's own preprocessed grid, annotated slices, 74 primary cases / 85 lesions (28 bright, 57 dark). Default policy = margin > 0; shifted policy = margin > t with t in {-0.5, ..., -4.0} chosen inside each outer training fold (5-fold by patient, fold = index in the sorted primary list mod 5) as the value with the smallest mean absolute sum-of-diameters error, applied to the held-out fold only. Lesions and the sum-of-diameters error as in the hackathon RECIST analysis (>= 10 mm, the two largest per case, detected = >= 10% covered, a miss counts 0 mm). Case-level bootstrap 95% CIs (per-arm seed). No readout re-fit (researcher's decision). Grid lesions matched to the lesion table: 0 diameter mismatches above 25% in either model.

| | A: default | A: shifted | D: default | D: shifted |
|---|---|---|---|---|
| median signed sum-of-diameters error (bias) | -31.4% | **-16.8%** | -34.0% | **-24.9%** |
| mean absolute error | 46.2% | 34.1% | 46.0% | 38.7% |
| cases with |error| >= 20% | 64.9% | 44.6% | 59.5% | 54.1% |
| **cases over-estimated >= 20% (the cost)** | 0.0% | **1.4%** | 1.4% | **1.4%** |
| mean tumour Dice (model grid) | 0.374 | 0.490 | 0.438 | 0.509 |
| detection, dark lesions | 84.2% | 91.2% | 86.0% | 91.2% |
| detection, bright lesions | 25.0% | 46.4% | 21.4% | 32.1% |
| **detection gap (dark minus bright)** | 0.592 | **0.448** | 0.645 | **0.591** |

| paired change (shifted minus default), point [95% CI] | A | D |
|---|---|---|
| detection gap | -0.144 [-0.311, +0.009] | -0.055 [-0.191, +0.062] |
| |bias| | **-0.146 [-0.188, -0.072]** | **-0.092 [-0.179, -0.039]** |
| over-estimation share >= 20% | +0.014 [0.000, +0.041] | 0.000 [0.000, 0.000] |

**Per-model verdicts (pre-registered rule: FIX GENERALISES needs BOTH the gap change (CI entirely below 0 and point <= -0.05) AND the |bias| change (CI entirely below 0 and point <= -0.05)):**
- **Model A: PARTIAL.** The bias criterion is met (-0.146, CI below 0); the gap criterion is not (point -0.144 clears the margin but the CI [-0.311, +0.009] includes 0).
- **Model D: PARTIAL.** The bias criterion is met (-0.092, CI below 0); the gap criterion is not (point -0.055 is only just past the margin and the CI [-0.191, +0.062] includes 0).
- **Over-estimation cost:** negligible in both (1.4% over-estimated >= 20% under the shift; no cost flag; VISTA reference 12.4%).

**Cross-model reading as fixed in the PREREG: "the fix generalises" only if BOTH A and D reach FIX GENERALISES. Neither does (both PARTIAL), so the fix does NOT generalise across both models in this form:** a threshold shift reduces the one-sided size bias in both models and raises bright-lesion detection (A 25% to 46%, D 21% to 32%) at almost no over-estimation cost, but it does not significantly close the bright-versus-dark detection gap in either model.

Limitations and facts to read with this (no rule changed): (1) **the chosen threshold sat at the lower edge of the grid (-4.0) in every outer fold for both models**, so the grid may be too narrow and the shift's effect understated; the pre-registered grid was kept. (2) Analysis on each model's own grid, not the original grid (A 1.5 mm, D 1.0 x 0.77 x 0.77 mm); the default policy is recomputed on the same grid so the comparisons are paired, but the default numbers differ slightly from Stage 1's original-grid figures. (3) 28 bright lesions give wide gap CIs; the gap criterion is the one that is hardest to meet with this cohort. (4) The bright and dark groups differ in size and other properties; the shift is a single global threshold.
