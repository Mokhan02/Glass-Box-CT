# W5 results: does bright-lesion contrast augmentation close the brightness gap? (PREREG.md with Amendment 1; **SINGLE-SEED: one training run per arm; training-run variance is not measured and the patient bootstrap does not cover it**)

Arms that ran (two instances; D0-seed2 and BRIGHT-25 were not run, decided by GPU count before any training, Amendment 1): **D0** = model D (no augmentation), **BRIGHT-50** (tumours shifted to liver + U(+10, +60) HU with probability 0.5), **SHAM-50** (liver + U(-60, -10) HU, probability 0.5). Same plan, fold 0, 1000 epochs, seed 20261020, nnU-Net 2.8.1 on one A10 each. HCC cohort 74 cases / 85 lesions (28 bright / 57 dark), the Stage 1 pipeline unchanged (the pipeline reproduces D0's stored Stage 1 numbers exactly), analysis `scripts/w5_analyze.py` (committed and hashed before any W5 result). Files: `results/w5_training/results/analysis/w5_results.json`, per-arm lesion tables, gates and W3 analyses under `results/w5_training/results/{D0,BRIGHT50,SHAM50}/`.

## Sanity gate (fold-0 Task03 validation, nnU-Net's own summary) and alignment gate
| arm | liver Dice (gate >= 0.93) | tumour Dice (gate >= 0.597) | gate | Amendment 2 alignment gate (HCC) |
|---|---|---|---|---|
| D0 (reference) | 0.9651 | 0.6466 | | OK |
| BRIGHT-50 | 0.9651 | 0.6779 | **PASS** | OK, 0 misaligned or failed |
| SHAM-50 | 0.9652 | 0.6880 | **PASS** | OK, 0 misaligned or failed |

## Per-arm gaps on the HCC cohort (dark minus bright; positive = bright segmented worse)
| arm | bright / dark detection | detection gap | bright / dark mean Dice | Dice gap | mean Dice (all lesions) | median signed sum-of-diameters error |
|---|---|---|---|---|---|---|
| D0 | 21.4% / 86.0% | +0.645 | 0.109 / 0.528 | +0.419 | 0.390 | -34.3% |
| BRIGHT-50 | **89.3%** / 98.2% | **+0.090** | 0.513 / 0.698 | **+0.185** | 0.637 | -9.9% |
| SHAM-50 | 35.7% / 87.7% | +0.520 | 0.148 / 0.580 | +0.431 | 0.437 | -25.5% |

The W3 per-arm labels (committed Stage 1 analysis, margin 0.10): D0 GAP (detection and Dice); **SHAM-50 GAP (detection and Dice)** (+0.520 [0.279, 0.719], +0.431 [0.314, 0.544]); **BRIGHT-50 GAP (Dice only)** (detection +0.090 [-0.032, 0.257], inconclusive on the margin; Dice +0.185 [0.034, 0.333]).

## Primary: the pre-registered contrasts (BRIGHT-50 minus comparator, gap change; paired case-level bootstrap, 1000 resamples)
| contrast | detection gap change [95% CI] | verdict | Dice gap change [95% CI] | verdict |
|---|---|---|---|---|
| BRIGHT-50 minus D0 | **-0.556 [-0.774, -0.343]** | CLOSES | **-0.234 [-0.367, -0.108]** | CLOSES |
| BRIGHT-50 minus SHAM-50 | **-0.430 [-0.656, -0.202]** | CLOSES | **-0.246 [-0.370, -0.120]** | CLOSES |

**Prediction ("BRIGHT-50 closes the gap compared with both D0 and SHAM-50"): CONFIRMED (detection and Dice), single seed.** All four cells meet the pre-registered rule (CI upper bound below 0 and point estimate at most -0.10). Two readings to keep with it: (1) the gap is reduced a lot, not eliminated: BRIGHT-50 still has a Dice gap of +0.185 (bright Dice 0.513 against dark 0.698) and its detection gap of +0.090 is inconclusive on the 0.10 margin; (2) the Dice contrast against D0 is the tightest cell (CI upper bound -0.108, just past zero).

## Secondary and exploratory
- **Sum-of-diameters bias:** median signed case error -34.3% (D0), -25.5% (SHAM-50), **-9.9% (BRIGHT-50)**; overall mean Dice 0.390, 0.437, 0.637.
- **EXPLORATORY, procedure effect (SHAM-50 minus D0, no verdict):** detection gap change -0.125 [-0.293, +0.015], Dice gap change +0.012 [-0.042, +0.059]. The procedure alone moved bright detection from 21% to 36% and left the Dice gap unchanged, with no CI excluding zero. Without D0-seed2 it is not possible to say how much of that is run-to-run variance.
- **Not run:** the D0-seed2 variance estimate, the BRIGHT-25 dose trend (Amendment 1), and M1/M2 on BRIGHT-50 (needs a GPU session and sign-off).

## Limits
Single seed per arm (every contrast is one BRIGHT-50 run against one D0 run and one SHAM-50 run; the bootstrap covers patient sampling only); one fold; 28 bright lesions; the synthetic brightening is a stand-in for real hypervascular lesions and acts inside the CT window (clipped at 197 HU), so the result shows that the training contrast distribution can move this gap in this recipe, not that real bright-lesion data would give the same gain; the HCC cohort mixes acquisitions and its contrast phase is unlabelled; W5 does not by itself identify what the model learned (M1/M2 on BRIGHT-50 are the planned follow-up). Reading rule from the W3 PREREG: this result is consistent with the training-data explanation for D's gap, and does not show causation in the clinical sense or for other models.

---

## Amendment 2: BRIGHT-50 replication (BRIGHT-50-s2, seed 20261028; added after the W5 results above were seen, as stated in PREREG Amendment 2)
Identical to BRIGHT-50 except the fixed seed; same gates, cohort, scorer, post pipeline and unchanged contrast analysis (`results/w5_training/results/analysis_s2/`, `.../replication/`, `.../BRIGHT50_s2/`). Early smoke PASS (seed 20261028 and BRIGHT p = 0.5 in the log, finite losses, epoch 2 took 59.5 s, 47.4% of 593 tumour samples edited, 1368 deltas all inside +10 to +60 HU; `BRIGHT50_s2/smoke_B50S2.json`). Sanity gate PASS (liver Dice 0.9647, tumour Dice 0.6732; BRIGHT-50 0.6779, D 0.6466). Alignment gate OK, 0 misaligned or failed.

| arm | bright / dark detection | detection gap | Dice gap | mean Dice | median signed sum-of-diameters error |
|---|---|---|---|---|---|
| D0 | 21.4% / 86.0% | +0.645 | +0.419 | 0.390 | -34.3% |
| SHAM-50 | 35.7% / 87.7% | +0.520 | +0.431 | 0.437 | -25.5% |
| BRIGHT-50 (seed 20261020) | 89.3% / 98.2% | +0.090 | +0.185 | 0.637 | -9.9% |
| **BRIGHT-50-s2 (seed 20261028)** | **85.7%** / 98.2% | **+0.125** | **+0.193** | 0.643 | -9.1% |

**Primary (unchanged W5 rule, BRIGHT-50-s2 as the BRIGHT arm):**
| contrast | detection gap change [95% CI] | verdict | Dice gap change [95% CI] | verdict |
|---|---|---|---|---|
| BRIGHT-50-s2 minus D0 | -0.520 [-0.738, -0.310] | CLOSES | -0.226 [-0.353, -0.106] | CLOSES |
| BRIGHT-50-s2 minus SHAM-50 | -0.395 [-0.618, -0.177] | CLOSES | -0.238 [-0.359, -0.114] | CLOSES |

**Prediction label: CONFIRMED (detection and Dice) again, so the BRIGHT-50 result REPLICATES under the pre-registered definition.** As in the first run, the gap is reduced a lot and not removed: the W3 per-arm label for BRIGHT-50-s2 is GAP (Dice only) (detection +0.125 [-0.016, 0.297], inconclusive on the 0.10 margin; Dice +0.193 [0.049, 0.338]), and the Dice contrast against D0 is again the tightest cell (CI upper bound -0.106).

**Descriptive seed variance for the augmented arm (not part of the verdict):** |detection gap difference| = 0.036, |Dice gap difference| = 0.008 between the two BRIGHT runs (bright detection 89.3% vs 85.7%, mean Dice 0.637 vs 0.643). **No D0-seed2 was run, so there is still no seed-variance estimate for D0 or SHAM-50, and the contrasts against them remain single-run on that side;** every reading stays single-seed per arm (two seeds for BRIGHT only). Limits as above; the replication was added after the first outcome was known, with its rule fixed before the new arm produced any result.
