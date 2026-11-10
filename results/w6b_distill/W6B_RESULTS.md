# W6b results (living file; research use only, no clinical claims)

Last updated 2026-11-10 05:20 EST. PREREG: `PREREG.md` (amended; lambda 143 and 573 set by the scale rule from training data; no WAW-TACE tuning). Frozen evaluation code; single runs, seed 20261110. Both full-network arms are reported, no selection between them. The forgetting guard (criterion 2) is not finished for either arm yet.

## Primary, per arm (PREREG section 6; an arm SUCCEEDS only if BOTH criteria hold)

| Arm | Criterion 1: bright arterial detection against BRIGHT (n 242), paired difference, 95% CI, margin -0.10 | Criterion 2: forgetting guard | Outcome |
|---|---|---|---|
| W6b-FULL-L143 (lambda 143) | 69.8% vs 77.3%: **-0.074 [-0.116, -0.037]: NOT SHOWN NON-INFERIOR** (the lower bound is below -0.10) | **FAILS only on drift: FORGETTING COST: drift.** Spleen, kidney, liver and false positives pass; 6 of 48 classes above the floor stay below 0.95 (below) | **NEITHER** (criterion 1 not shown, criterion 2 fails) |
| W6b-FULL-L573 (lambda 573) | 69.0% vs 77.3%: **-0.083 [-0.121, -0.046]: NOT SHOWN NON-INFERIOR** (the lower bound is below -0.10) | **FAILS only on drift: FORGETTING COST: drift.** Spleen, kidney, liver and false positives pass; 11 of 48 classes above the floor stay below 0.95 (below) | **NEITHER** (criterion 1 not shown, criterion 2 fails) |
| W6b-DEC-L143 (decoder only, lambda 143) | 59.9% vs 77.3%: **-0.174 [-0.226, -0.124]: NOT SHOWN NON-INFERIOR** | **FAILS only on drift: FORGETTING COST: drift.** Spleen, kidney, liver and false positives pass; 1 of 48 classes above the floor stays below 0.95 (gallbladder 0.920) | **NEITHER** (criterion 1 not shown, criterion 2 fails on one class) |

FULL-L143 guard in detail (against the baseline, same cases as W6; `guard/forget_analysis_W6b-FULL-L143_stream.json`): spleen 0.9609 (change -0.0015 [-0.0020, -0.0010], pass); **kidney Dice 0.8784 (change +0.0041 [+0.0010, +0.0083], pass; BRIGHT: -0.056)**; liver 0.9685 (change -0.0007, pass); false positives median 0.46 mL against 0.05 baseline (pass); kidney-tumour side check 0.5021 (baseline 0.5165, BRIGHT 0.4941); drift: **6 of the 48 classes above the 10 mL floor fail the 0.95 rule (BRIGHT: 32)**: gallbladder 0.918, vertebrae T10 0.909 and T9 0.917, left ribs 9 and 10 0.947 and 0.948, right rib 8 0.914 (lowest class mean Dice 0.909; BRIGHT's lowest 0.748). The strict guard therefore still reports FORGETTING COST: drift, but almost all of the forgetting seen in W6 is gone. **Graded secondaries (`scripts/w6b_secondary.py`, 95% bootstrap CIs, B 2,000; `guard/w6b_secondary_W6b-FULL-L143.json`):** failing drift classes 6 [2, 9] against BRIGHT's 32 [28, 37]; paired difference -26 [-31, -22]; kidney Dice change against BRIGHT +0.060 [+0.051, +0.071]; against the baseline +0.004 [+0.001, +0.008] (BRIGHT: -0.056 [-0.065, -0.048]).

FULL-L573 guard in detail (`guard/forget_analysis_W6b-FULL-L573_stream.json`): spleen 0.9613 (change -0.0011, pass); kidney Dice 0.8689 (change -0.0053 [-0.0075, -0.0028], pass: inside the 0.02 and -0.03 limits); liver 0.9690 (-0.0003, pass); false positives median 0.33 mL against 0.05 baseline (pass); kidney-tumour side check 0.5210 (baseline 0.5165); drift: **11 of the 48 classes above the floor fail 0.95 (L143: 6; BRIGHT: 32)**: vertebrae L5 0.936, L2 0.947, T11 0.934, T10 0.810, T9 0.779; left ribs 8, 9, 10 0.913, 0.912, 0.943; right ribs 8, 9, 10 0.896, 0.921, 0.943; lowest class mean Dice 0.779 (L143: 0.909). The stronger leash (lambda 573) did not reduce drift further than lambda 143 here: more classes fail (11 against 6) and the worst class is lower, although its kidney Dice is close to the baseline (-0.005 against +0.004). Single runs; the differences between the two lambdas are not tested.

**FULL-L573 graded secondaries (`guard/w6b_secondary_W6b-FULL-L573.json`):** failing drift classes 11 [5, 15] against BRIGHT's 32 [28, 37]; paired difference -21 [-27, -18]; kidney Dice change against BRIGHT +0.051 [+0.043, +0.060]; against the baseline -0.005 [-0.008, -0.003].

The decoder-only arm (DEC-L143) loses more: 59.9% against 77.3% (its W6 counterpart BRIGHT-DEC was 67.8%), -0.174 [-0.226, -0.124], so it also fails criterion 1, by a wider margin; its guard is pending.

DEC-L143 guard in detail (`guard/forget_analysis_W6b-DEC-L143_stream.json`): spleen 0.9612 (change -0.0012, pass); kidney Dice 0.8732 (change -0.0010 [-0.0022, +0.0001], pass; BRIGHT-DEC in W6: -0.030); liver 0.9687 (-0.0005, pass); false positives median 0.26 mL against 0.05 baseline (pass); kidney-tumour side check 0.5165, identical to the baseline as for BRIGHT-DEC (the decoder-only arm cannot change the point route); drift: **1 of the 48 classes above the floor fails 0.95: gallbladder 0.920 (BRIGHT-DEC in W6: 20; lowest class mean Dice 0.920 against 0.785)**. The decoder-only arm with distillation is therefore the cleanest on forgetting (one class short of a full guard pass) and the weakest on bright-tumour detection (59.9%). Graded secondaries with CIs follow from `scripts/w6b_secondary.py`.

Both full-network arms fail criterion 1 under the strict rule written in the PREREG: the point estimates (-0.074 and -0.083) are inside the 0.10 margin, but the CI lower bounds (-0.116 and -0.121) are not above -0.10. So whatever the guard shows, neither arm can be labelled SUCCEEDS; the best possible outcome is GUARD ONLY (guard passes, detection non-inferiority not shown). Reference noise from the PREREG: BRIGHT seed 2 minus BRIGHT = -0.008 [-0.036, +0.016]; W6 BRIGHT-DEC minus BRIGHT = -0.095 [-0.137, -0.056]. The strict rule was kept as agreed; no margin or rule was changed after seeing these numbers. File: `primary_criterion1.json`; script `scripts/w6b_primary.py` (hashed; DEVIATIONS 36).

## W4b labels and detection by contrast group

| Model | theta [95% CI] | Label | Arterial dark (n 55) | Arterial bright (n 242) | Portal bright (n 166) | Portal dark (n 131) |
|---|---|---|---|---|---|---|
| BASELINE (already seen) | +0.159 [+0.036, +0.280] | INCONCLUSIVE | 56.4% | 21.5% | 21.1% | 69.5% |
| SHAM | +0.245 [+0.121, +0.379] | BRIGHT PENALTY | 80.0% | 21.5% | 27.1% | 78.6% |
| BRIGHT | -0.032 [-0.164, +0.100] | INCONCLUSIVE | 74.5% | 77.3% | 65.7% | 78.6% |
| W6b-FULL-L143 | +0.031 [-0.119, +0.189] | INCONCLUSIVE | 78.2% (43) | 69.8% (169) | 56.0% (93) | 79.4% (104) |
| W6b-FULL-L573 | -0.036 [-0.179, +0.119] | INCONCLUSIVE | 69.1% (38) | 69.0% (167) | 55.4% (92) | 75.6% (99) |
| W6b-DEC-L143 | +0.037 [-0.107, +0.201] | INCONCLUSIVE | 67.3% (37) | 59.9% (145) | 47.6% (79) | 76.3% (100) |

Both W6b arms keep most of the bright-tumour gain (about 70% of arterial-bright tumours detected against 21.5% at baseline) but lose about 8 points against BRIGHT (77.3%); their theta values are INCONCLUSIVE like BRIGHT's, with the margin-0.25 equivalence reading NO PENALTY as for BRIGHT. Full analysis files: `analysis_full/` (all arms, frozen script) and `detection_table_w6b.json`.

## Pending

Forgetting guard for both arms (organs, side check, false positives, drift of the 123-class list), HCC-TACE-Seg, graded secondaries (`scripts/w6b_secondary.py`: failing-class count against BRIGHT's 32, kidney Dice change against BRIGHT, both with CIs), and DEC-L143 (starts on instance 1 after FULL-L143's evaluation). Training notes (FULL-L143): mean KL to the baseline over training 9.2e-5 against 8.7e-4 for BRIGHT's final model; mean task loss 0.527 (BRIGHT 0.52).
