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

---

## EXPLORATORY: extended threshold grid (not pre-registered; the verdicts above stand)

**The pre-registered verdicts (A PARTIAL, D PARTIAL; the fix does NOT generalise) are unchanged.** This section is an exploratory re-run requested after the grid-edge finding. Nothing here replaces, re-labels or overrides them.

Why the grid was suspect: the M1 head margins (A -6.3, D -7.1, from M1) predicted that the pre-registered grid (-0.5 to -4.0) was too narrow, and in the pre-registered run every outer fold chose the edge (-4.0).

Method: identical to the pre-registered M4 (same margins, cases, lesions, outcomes, 5-fold patient-nested CV, per-arm bootstrap seeds, no readout re-fit), with the grid extended to -10.0 in the same 0.5 step (`--grid-min`, `t_grid` in `w3s2_m4.py`; defaults unchanged). Files: `m4/A_extended/m4_results_extended.json`, `m4/D_extended/m4_results_extended.json`.

| | A: default | A: shifted (ext.) | D: default | D: shifted (ext.) |
|---|---|---|---|---|
| chosen thresholds, folds 0-4 | | -7.5, -7.5, -7.5, -6.5, -7.5 | | -7.5 in all five folds |
| median signed sum-of-diameters error (bias) | -31.4% | **-1.3%** | -34.0% | **-6.3%** |
| mean absolute error | 46.2% | 27.0% | 46.0% | 31.0% |
| cases with |error| >= 20% | 64.9% | 37.8% | 59.5% | 41.9% |
| **cases over-estimated >= 20% (the cost)** | 0.0% | **12.2%** | 1.4% | **10.8%** |
| mean tumour Dice | 0.374 | 0.558 | 0.438 | 0.568 |
| detection, dark | 84.2% | 96.5% | 86.0% | 98.2% |
| detection, bright | 25.0% | 71.4% | 21.4% | 39.3% |
| **detection gap** | 0.592 | **0.251** | 0.645 | **0.590** |

| paired change (shifted minus default), point [95% CI] | A | D |
|---|---|---|
| detection gap | **-0.341 [-0.559, -0.146]** | -0.056 [-0.235, +0.103] |
| |bias| | **-0.302 [-0.354, -0.208]** | **-0.278 [-0.359, -0.144]** |
| over-estimation share >= 20% | +0.122 [+0.054, +0.203] | +0.095 [+0.041, +0.176] |

Exploratory verdicts under the same rule: **A: FIX GENERALISES** (both criteria met), with the over-estimation cost flagged (+12.2 points, above the +0.10 flag). **D: PARTIAL** (bias criterion met; gap criterion not, point -0.056 with CI spanning 0), cost not flagged (+9.5 points, just under the flag).

Exploratory cross-model reading: the chosen thresholds (A about -7.5, D -7.5 in all folds) sit close to the M1 head margins (-6.3, -7.1), consistent with the prediction that the pre-registered grid could not reach the useful operating point. With the wider grid the bias is almost fully removed in both models. The fix would still **not** generalise across both models by the PREREG rule: D's bright-lesion detection rises (21% to 39%) but the detection gap does not close significantly, and the shift costs about 10-12 points of over-estimation in both models. EXPLORATORY: the grid was widened after seeing the pre-registered result (the edge finding), so the extended thresholds and these contrasts are not confirmatory.
