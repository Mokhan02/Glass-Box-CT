# Pre-registration W2 v3: Dice gap on the unseen main set

Status: APPROVED (2026-10-17) with two edits (the HCC 26/1 arm is dropped from the main run; the analysis script is committed before the GPU run), committed on `research/roadmap` BEFORE any main-set lesion was run with the v3 arms. Deviations go to `results/w2_kidney/DEVIATIONS.md`. **Inclusion or exclusion changes need the researcher's sign-off before they are applied, even pre-analysis.**

## History, stated plainly
- **v1 was invalid.** It sent point label 1 for every arm and used the legacy classes 129 and 130; class 26 is a special class that needs point label 3, and 129/130 are unsupported (`PREREG_v2.md`, `validity_check.md`). v1's predictions were organ-sized, so its coverage gaps do not measure tumour under-segmentation.
- **The v2 pilot gate failed for all three primary arms.** The v2 gate was set before the pilot, passed on none of the three primary arms, and by the pre-agreed rule v2 stopped after the pilot (no main run).
- **Bright lesions failed the v2 gate more often than dark lesions (3 to 4 of 5 vs 0 to 1 of 5 per arm: kidney 133/1 3 vs 1, HCC 26/3 3 vs 1, HCC 133/1 4 vs 0).** This is exploratory (n = 10 lesions per organ, counted after the pilot). It means the v2 gate was correlated with the outcome under study, so it cannot serve as a validity gate for the main analysis; that is why v3 replaces it.
- **v3 was written after the v2 pilot and BEFORE any main-set lesion was run with the v3 arms.** The main set is unseen with respect to these arms. Two honest limits: the v1 arms (different, invalid prompts) were run on the main-set lesions and their results were seen (for example, bright kidney lesions were covered more in v1), so my expectations are not blank; and the Dice choice, the failure-mode definitions and the dark-lesion gate were all chosen after seeing the pilot.

## Question
Given where a tumour is, is the point-prompted segmentation worse (lower Dice) for tumours brighter than the surrounding parenchyma, in kidney (C4KC-KiTS) as in liver (HCC)? Point prompts give location, so this is appearance-driven segmentation quality given location, not detection. The zero-shot arms (label 133) carry no tumour class and test a generic point-based segmenter; the HCC 26/3 arm is the supported tumour-class route and has no kidney counterpart.

## Data
C4KC-KiTS: 210 cases, 219 primary lesions (two largest >= 10 mm per case; 53 bright, 166 dark); HCC-TACE-Seg UID route: 75 cases, 86 lesions (28 bright, 58 dark); same lesion definition and contrast function as W1/W2 (`glassbox/audit/contrast.py`). Bright = lesion mean HU minus parenchyma-shell mean HU > 0. **Main set = all primary lesions except the 20 v2 pilot lesions (`v2_pilot_selection.csv`): kidney 209 lesions (48 bright, 161 dark), HCC 76 lesions (23 bright, 53 dark).** Every C4KC-KiTS output carries "possibly seen in training (images, via AbdomenCT-1K)".

## Arms (unchanged from v2; single object, B = 1, one positive point at the interior point, point label as in `glassbox/seg/w2_arms.py`)
| Arm | label_prompt | point label | Role |
|---|---|---|---|
| Kidney zero-shot | 133 | 1 | primary |
| HCC supported tumour class | 26 | 3 | primary (reported separately from the like-for-like comparison) |
| HCC zero-shot (matched) | 133 | 1 | primary |
| ~~HCC v1-equivalent~~ | 26 | 1 | **dropped from the main run** (see the validity gate: it failed, 1 of 5 dark pilot lesions pass, so it cannot contribute; it exists only in the v2 pilot data) |
The runner asserts on every call that the point label reaching the point head equals the arm's label. Prediction = probability > 0.5 on the original grid, annotated slices only. All predicted masks are saved. The main run covers the three primary arms only (kidney 133/1, HCC 26/3, HCC 133/1).

## Validity gate (replaces the v2 gate; set AFTER seeing the pilot)
On the DARK pilot lesions only (5 per organ), an arm is valid if >= 80% of them pass the v2 per-lesion criterion (predicted/lesion volume < 3x AND share of predicted voxels inside the lesion > 0.3). **The pilot already passes it: dark pilot lesions passing = kidney 133/1 4 of 5 (80%), HCC 26/3 4 of 5 (80%), HCC 133/1 5 of 5 (100%); the secondary arm 26/1 passes 1 of 5 (20%) and is not valid.** Stated limits: the rule was set after seeing the pilot; two primary arms sit exactly on the 80% threshold (one more dark failure would have flipped them); the gate only shows the arm yields lesion-scale masks for dark lesions, it cannot validate bright lesions (their performance is the outcome). No further gating happens before the main run. **Because the 26/1 arm failed this gate, it is not run on the main set and does not contribute to any v3 result;** the main run covers the three primary arms only.

## Primary analysis (main-set lesions only; the pilot lesions are excluded)
**Dice gap = mean Dice in dark lesions minus mean Dice in bright lesions** (Dice between the ground-truth lesion and the whole prediction on annotated slices, stored as `dice_full`), per primary arm, with a patient-level (case) bootstrap 95% CI (1000 resamples, seed 0, percentile). Verdict per arm, in order: **GAP** if CI lower > 0 and point >= 0.10; **NO MEANINGFUL GAP** if CI upper < 0.10; **INCONCLUSIVE** otherwise. **Like-for-like replication = kidney 133/1 and HCC 133/1 both GAP.** The HCC 26/3 verdict is reported separately as the HCC tumour-class result. Three primary verdicts are reported without multiplicity correction. The kidney-minus-HCC (matched arms) gap difference with CI is descriptive only.

## Key secondaries (reported with the primary, never instead of it)
(a) **Failure-mode rates by brightness.** Per lesion, mutually exclusive, in this order: **LEAK** = predicted/lesion volume >= 3x OR inside-lesion share <= 0.3; else **UNDER** = coverage < 0.5; else **OK**. Rates by brightness and the bright-minus-dark difference in the LEAK rate and in the UNDER rate, with case-level bootstrap CIs.
(b) **Size-adjusted:** OLS of Dice on `bright` and log(diameter) with a patient-clustered bootstrap CI (1000 resamples, seed 0); the adjusted gap (minus the `bright` coefficient) is stated next to the headline, and **if the adjusted effect disappears (CI includes 0 or has the opposite sign to the headline gap) that is stated beside it.** The gap within strata <30, 30 to 60 and >60 mm is reported descriptively.
(c) **Coverage gap** (dark minus bright, share of ground-truth lesion voxels predicted) for continuity with v1 and v2.
(d) **Sensitivity including the pilot lesions:** the primary, (a), (b) and (c) are re-run on all primary lesions (kidney 219, HCC 86), using the pilot predictions of the three primary arms already made with the same code. The primary remains the main-set-only analysis.
Also reported per arm, split by bright and dark: coverage, Dice, Dice within 10 mm, and the predicted/GT diameter ratio.

## Analysis script (committed before the GPU run)
`scripts/w2v3_analyze.py`, commit **fa4a639**, git blob sha1 `04a6a488e67bc795eedd8eeb0541766f11b3faa2`, sha256 `70d0e4c92fc9eded44f1e41ebcc3cb90a7c49fdaa417a009778e5fe6f2305ed0`. Tests: `tests/test_w2v3_analyze.py` (failure-mode order, verdict rule order, a synthetic Dice gap and null, the sign handling of the size-adjusted flag, an end-to-end synthetic run). It was run end to end on the v2 PILOT predictions as a smoke test only (a pipeline check, never a result; those numbers are not reported or interpreted). The same file is run unchanged on the main-set predictions after verifying its hash against the above.

## Confounds (as v2, repeated in the write-up)
Possible in-distribution images (C4KC-KiTS via AbdomenCT-1K; tumour labelling there unknown); arterial/corticomedullary-phase KiTS (bright cortex inside the parenchyma shell); point prompts give location; contrast phase and tumour type differ between kidney and HCC; `Mass` may contain cysts; size is related to brightness; the zero-shot route carries no tumour class; the validity gate and the Dice choice were set after seeing the pilot; Dice mixes leakage and under-segmentation (secondary (a) separates them); the HCC bright group has only 23 main-set lesions, so its CI will be wide.

## Mechanism steps
Not part of v3.

## Procedure after approval
1. Commit this file with the script hash (done before any run). 2. The instance stays up. 3. Run the three primary arms on the main set with all masks saved (`scripts/w2v2_run.py main ... --arms kidney_133_1,hcc_26_3,hcc_133_1`). 4. Verify the script hash, run `scripts/w2v3_analyze.py` unchanged, and stop at the verdicts. 5. Report, with the GPU idle times.
