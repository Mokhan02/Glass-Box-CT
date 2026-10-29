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


---

## Amendment 3: mechanism after the fix (BRIGHT-50, seed 20261020; PREREG Amendment 3 approved 2026-10-28; run 2026-10-28 18:49 to 2026-10-29 07:41, America/New_York)
Question: does bright-tumour augmentation remove the brightness dependence of the failure and of the tumour concept? Model BRIGHT-50 (seed 20261020; the replication model is not used), hook `encoder.stages[3]` (as D0), HCC cohort 74 primary cases (75 run), the committed W3 Stage 2 runners with two default-preserving overrides (trainer class and reference table; DEVIATIONS 15), nnU-Net default inference. Gates all passed: checkpoint sha256 and validation Dice equal to the W5 sanity gate (0.9651 / 0.6779); hook check (propagation to both consumers; unhooked output reproduces the W5 BRIGHT-50 table, max coverage difference 4e-5); M3 validated 75 of 75 cases, 204 rows, 0 problems; M2 full run base versus W5 table within 5e-5 on every case, 0 detection-flag mismatches, identity hook equal to base. The 1.25 x time rule never fired (M3 4.7 h against a 5.0 h limit, extraction 1.9 h, M2 5.6 h; total 12 h 52 min, corrected estimates in DEVIATIONS 15). Summary files and the driver log: `results/w5_training/a3/`.

### Primary: M3 contrast BRIGHT-50 minus D0 (Arm A, dark detected lesions edited to bright): **REDUCED**
Delta_A = mean over the 49 lesions detected unedited by both models (46 cases) of [D_A(BRIGHT-50) - D_A(D0)], D_A = coverage change of the edit minus that of the sign-keeping control; case-level bootstrap, margin 0.10 (pre-registered rule: CI lower bound > 0 and point >= +0.10).

| | D0 | BRIGHT-50 |
|---|---|---|
| Arm A lesions detected unedited (all 49 of D0's are among BRIGHT-50's) | 49 | 56 |
| median change in coverage when brightened | -0.454 [-0.608, -0.270] | +0.000 [-0.014, +0.013] |
| median D_A (edit minus control) | -0.336 [-0.542, -0.185] | -0.006 [-0.045, +0.147] |
| sign test (direction as hypothesised) | 40/48, p = 3.3e-06 | 30/56, p = 0.69 |
| residual Arm A harm | PRESENT | ABSENT |

**Delta_A = +0.510 [+0.441, +0.572] -> REDUCED.** Descriptive companions (no verdict): change in coverage by the edit alone +0.429 [+0.347, +0.505]; lesion Dice D_A +0.541 [+0.478, +0.608]; unpaired difference of medians on each model's own lesion set +0.330 [+0.194, +0.521]; excluding flagged edits (n = 39) +0.488 [+0.409, +0.566], also REDUCED. **Reading the size:** the mean D_A on the common set is -0.378 in D0 and +0.132 in BRIGHT-50, above zero because the sign-keeping control (making the lesion darker) now lowers coverage in BRIGHT-50 while brightening does nothing; the "relative reduction 135%" printed by the script is therefore not a share of harm removed. The result is "the brightening harm is gone", not "more than removed".
**Arm B (bright to dark), no verdict (3 lesions against the required 10; D0 had 22).** BRIGHT-50 baseline coverage 0.02, 0.03, 0.05; darkening changes coverage by -0.007, -0.022, +0.024 (D0, same lesions: +0.000, +0.000, +0.031); the control that makes the lesion brighter changes it by +0.004, +0.348, +0.415 (D_B -0.011, -0.370, -0.390; D0 D_B +0.000, +0.000, +0.058). In two of three, brighter now means more detection, the opposite of D0's pattern; n = 3, a pointer and not a finding. The full Stage 2 M3 verdict is INCONCLUSIVE for BRIGHT-50 by construction (stated in the amendment) and is not used.

### Secondary: M2 (cross-fitted rank-1 LEACE at stage 3): **NOT NECESSARY** (no flags)
| | D0 (Stage 2) | BRIGHT-50 |
|---|---|---|
| lesions detected unedited | | 81 in 71 cases |
| lesion Dice change, tumour direction erased (LEACE) | -0.121 [-0.150, -0.095] | **-0.042 [-0.056, -0.030]** |
| random direction | -0.001 | +0.001 [+0.000, +0.002] |
| local-HU (brightness) direction | +0.006 | **-0.042 [-0.072, -0.016]** |
| LEACE minus random | -0.120 | -0.043 [-0.057, -0.030] |
| LEACE minus HU | -0.127 | **+0.000 [-0.022, +0.024]** |
| still detected after LEACE | 85.5% | 100% |
| liver Dice change | -0.0028 | -0.002 [-0.004, -0.000] |
| post-erasure probe AUROC (must be <= 0.60 at the upper bound) | 0.535 [0.500, 0.584] | 0.517 [0.480, 0.581] (before erasure 0.910 [0.883, 0.934]) |
| verdict | USED | **NOT NECESSARY** |

The pre-registered rule needs LEACE minus the HU control to have a CI entirely below zero; it does not (+0.000 [-0.022, +0.024]), the other conditions hold, no flags (liver guard and the post-erasure probe check pass), so the run does test necessity and the verdict is NOT NECESSARY. **Reading, as stated in the amendment before the run:** NOT NECESSARY does not show the concept is unused (one linear direction at one layer). What it does show is descriptive: after the fix the same erasure costs about a third of what it cost D0 (-0.042 against -0.121), is no larger than erasing a brightness-only direction (which in D0 did nothing), and leaves every detected lesion detected. That is consistent with the tumour direction at this layer having become entangled with brightness, but this was not tested and should not be read as shown. EXPLORATORY split (no verdict): bright lesions (25 in 23 cases) LEACE -0.039 [-0.065, -0.016], HU -0.056 [-0.127, -0.000]; dark lesions (56 in 52 cases) LEACE -0.044 [-0.059, -0.029], HU -0.036 [-0.069, -0.012]: the same size in both.

### EXPLORATORY: paired projection (no verdict)
Arm A, 56 lesions: mean change of the lesion's mean token along BRIGHT-50's own cross-fitted tumour direction -0.015 (D0 -0.04); Spearman correlation of that change with the coverage change **0.35 [0.05, 0.60]** (D0 0.78 [0.64, 0.87]). The coupling between movement along the tumour direction and loss of detection is much weaker after the fix.

### M1
Not run as a verdict (stated in advance): BRIGHT-50 misses 3 of 28 bright lesions (the replication model 4) against 22 in D0, too few for a missed-versus-detected probe.

### Limits and confounds
1. **The M3 edit and the training augmentation are the same kind of manipulation** (constant HU offset of the lesion towards or beyond the liver, +10 to +60 HU in training). REDUCED shows reduced sensitivity to this manipulation; it cannot separate "no longer depends on brightness" from "learned to be invariant to this edit". The behavioural W5 result on real lesions (bright detection 21.4% to 89.3%, replicated 85.7%) is the independent evidence.
2. **One training run** for BRIGHT-50 (the replication run is not examined mechanistically) and one for D0, which has no seed-variance estimate: part of any model difference is unmeasured run-to-run variance.
3. One fold, one cohort, one disease, contrast phase unlabelled; 49 common Arm A lesions give wide intervals; constant-offset edits change mean contrast only; the controls also move HU; no multiplicity correction (one primary contrast); BRIGHT-50 detects 57 of 58 dark lesions unedited, so Arm A starts from a high baseline.
4. M2 and the projection concern one linear direction at one layer (stage 3, 16 mm through-plane token); a null (NOT NECESSARY) is not evidence that the concept is unused.
5. **Records:** the per-lesion tables (M3 `m3_lesions.csv`, M2 `lesions.csv` and `cases.csv`, token tables, margin maps, erasers) are on the persistent filesystem under the run folder and were not copied into the repository before the instance was terminated; the repository holds the analysis outputs (`m3_contrast.*`, `a3_m2_analysis`, `a3_m3_analysis`, `a3_m2_by_brightness`) and the driver log. Code and hashes: `results/w5_training/code_hashes_a3.txt`.
