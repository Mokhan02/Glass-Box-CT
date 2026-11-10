# W6b results (living file; research use only, no clinical claims)

Last updated 2026-11-09 23:05 EST. PREREG: `PREREG.md` (amended; lambda 143 and 573 set by the scale rule from training data; no WAW-TACE tuning). Frozen evaluation code; single runs, seed 20261110. Both full-network arms are reported, no selection between them. The forgetting guard (criterion 2) is not finished for either arm yet.

## Primary, per arm (PREREG section 6; an arm SUCCEEDS only if BOTH criteria hold)

| Arm | Criterion 1: bright arterial detection against BRIGHT (n 242), paired difference, 95% CI, margin -0.10 | Criterion 2: forgetting guard | Outcome |
|---|---|---|---|
| W6b-FULL-L143 (lambda 143) | 69.8% vs 77.3%: **-0.074 [-0.116, -0.037]: NOT SHOWN NON-INFERIOR** (the lower bound is below -0.10) | **FAILS only on drift: FORGETTING COST: drift.** Spleen, kidney, liver and false positives pass; 6 of 48 classes above the floor stay below 0.95 (below) | **NEITHER** (criterion 1 not shown, criterion 2 fails) |
| W6b-FULL-L573 (lambda 573) | 69.0% vs 77.3%: **-0.083 [-0.121, -0.046]: NOT SHOWN NON-INFERIOR** (the lower bound is below -0.10) | pending (evaluation not started) | cannot be SUCCEEDS |

FULL-L143 guard in detail (against the baseline, same cases as W6; `guard/forget_analysis_W6b-FULL-L143_stream.json`): spleen 0.9609 (change -0.0015 [-0.0020, -0.0010], pass); **kidney Dice 0.8784 (change +0.0041 [+0.0010, +0.0083], pass; BRIGHT: -0.056)**; liver 0.9685 (change -0.0007, pass); false positives median 0.46 mL against 0.05 baseline (pass); kidney-tumour side check 0.5021 (baseline 0.5165, BRIGHT 0.4941); drift: **6 of the 48 classes above the 10 mL floor fail the 0.95 rule (BRIGHT: 32)**: gallbladder 0.918, vertebrae T10 0.909 and T9 0.917, left ribs 9 and 10 0.947 and 0.948, right rib 8 0.914 (lowest class mean Dice 0.909; BRIGHT's lowest 0.748). The strict guard therefore still reports FORGETTING COST: drift, but almost all of the forgetting seen in W6 is gone. **Graded secondaries (`scripts/w6b_secondary.py`, 95% bootstrap CIs, B 2,000; `guard/w6b_secondary_W6b-FULL-L143.json`):** failing drift classes 6 [2, 9] against BRIGHT's 32 [28, 37]; paired difference -26 [-31, -22]; kidney Dice change against BRIGHT +0.060 [+0.051, +0.071]; against the baseline +0.004 [+0.001, +0.008] (BRIGHT: -0.056 [-0.065, -0.048]).

Both arms fail criterion 1 under the strict rule written in the PREREG: the point estimates (-0.074 and -0.083) are inside the 0.10 margin, but the CI lower bounds (-0.116 and -0.121) are not above -0.10. So whatever the guard shows, neither arm can be labelled SUCCEEDS; the best possible outcome is GUARD ONLY (guard passes, detection non-inferiority not shown). Reference noise from the PREREG: BRIGHT seed 2 minus BRIGHT = -0.008 [-0.036, +0.016]; W6 BRIGHT-DEC minus BRIGHT = -0.095 [-0.137, -0.056]. The strict rule was kept as agreed; no margin or rule was changed after seeing these numbers. File: `primary_criterion1.json`; script `scripts/w6b_primary.py` (hashed; DEVIATIONS 36).

## W4b labels and detection by contrast group

| Model | theta [95% CI] | Label | Arterial dark (n 55) | Arterial bright (n 242) | Portal bright (n 166) | Portal dark (n 131) |
|---|---|---|---|---|---|---|
| BASELINE (already seen) | +0.159 [+0.036, +0.280] | INCONCLUSIVE | 56.4% | 21.5% | 21.1% | 69.5% |
| SHAM | +0.245 [+0.121, +0.379] | BRIGHT PENALTY | 80.0% | 21.5% | 27.1% | 78.6% |
| BRIGHT | -0.032 [-0.164, +0.100] | INCONCLUSIVE | 74.5% | 77.3% | 65.7% | 78.6% |
| W6b-FULL-L143 | +0.031 [-0.119, +0.189] | INCONCLUSIVE | 78.2% (43) | 69.8% (169) | 56.0% (93) | 79.4% (104) |
| W6b-FULL-L573 | -0.036 [-0.179, +0.119] | INCONCLUSIVE | 69.1% (38) | 69.0% (167) | 55.4% (92) | 75.6% (99) |

Both W6b arms keep most of the bright-tumour gain (about 70% of arterial-bright tumours detected against 21.5% at baseline) but lose about 8 points against BRIGHT (77.3%); their theta values are INCONCLUSIVE like BRIGHT's, with the margin-0.25 equivalence reading NO PENALTY as for BRIGHT. Full analysis files: `analysis_full/` (all arms, frozen script) and `detection_table_w6b.json`.

## Pending

Forgetting guard for both arms (organs, side check, false positives, drift of the 123-class list), HCC-TACE-Seg, graded secondaries (`scripts/w6b_secondary.py`: failing-class count against BRIGHT's 32, kidney Dice change against BRIGHT, both with CIs), and DEC-L143 (starts on instance 1 after FULL-L143's evaluation). Training notes (FULL-L143): mean KL to the baseline over training 9.2e-5 against 8.7e-4 for BRIGHT's final model; mean task loss 0.527 (BRIGHT 0.52).
