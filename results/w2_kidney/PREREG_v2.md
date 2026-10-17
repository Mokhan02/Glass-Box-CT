# Pre-registration W2 v2: kidney replication with supported point routes

Status: APPROVED (2026-10-16) and committed on `research/roadmap` BEFORE any v2 run, with the pilot selection file. Deviations go to `results/w2_kidney/DEVIATIONS.md`. **Inclusion or exclusion changes need the researcher's sign-off before they are applied, even pre-analysis.**

## Why v2 exists, stated plainly
- **v2 was designed after v1's results were seen.** It is not an independent pre-registration; it is a corrected protocol written in response to v1's outcome. v1's results stay in the log, unchanged, marked invalid for the question they were meant to answer.
- **v1 used class prompts the model does not support.** v1 prompted class 129 (kidney mass) and 130 (liver tumor) with a point. The MONAI VISTA-3D docs (`docs/inference.md`) call 129 ("insufficient data or dataset excluded") and 130 ("already have hepatic tumor") legacy classes kept "due to legacy issue", and say a zero-shot class must use `label_prompt` between 133 and 254 with points, B = 1 for point prompts. v1's predictions were organ-sized (16-lesion audit, `validity_check.md`), so v1's coverage gap does not measure tumour under-segmentation.
- **The class and point-label issue was found after v1.** v1 sent `point_labels = 1` for every arm, including class 26. In MONAI, class 26 is a "special" class (`SPECIAL_INDEX = (23, 24, 25, 26, 27, 57, 128)`, `monai/apps/vista3d/sampler.py` line 24): for special classes the point label must be 3 (positive) or 2 (negative), which add `special_class_embed` to the point embedding (`vista3d.py` lines 545 and 546), and the inferer docs say "2/3 means negative/positive points for special supported classes (e.g. tumor, vessel)" (`monai/apps/vista3d/inferer.py` line 59). The conversion 1->3 / 0->2 exists only in training and validation sampling (`_get_point_label`, sampler line 128; `convert_point_label`, defined at `vista3d.py` line 120 and called only from `sample_points_patch_val`, line 179) and is NOT applied to points a user supplies at inference. The vendored pipeline passes user point labels through unchanged (it accepts -1, 0, 1, 2, 3 as "special flags", `vista3d_pipeline.py` lines 272 to 274, and contains none of these names). So v1's class-26 + point arm used the wrong point label, and its organ-sized result is not evidence about class 26.
- **v1's class 26 and class 130 arms were nearly the same:** per lesion, predicted diameters identical in 81 of 86 lesions, coverage correlation 0.9996, Dice within 10 mm correlation 0.9999, and all three metrics exactly identical in only 2 lesions; the class prompt (26 vs 130) had almost no effect and the point dominated.
- **Not established:** that class 26 + point label 3 or label 133 + point label 1 gives lesion-scale masks. The pilot gate below tests that, and it may fail.

## Question
Given where a tumour is, is coverage lower for tumours brighter than the surrounding parenchyma, in kidney (C4KC-KiTS) as in liver (HCC), under supported point routes? The zero-shot route (133) carries no tumour class, so it tests a generic point-based segmenter, NOT the class-26 tumour concept; the class 26 + point label 3 arm is the supported tumour-class route for HCC and has no kidney counterpart. Point prompts give location, so this is appearance-driven under-segmentation given location, not detection.

## Data
Same as v1: C4KC-KiTS, 210 cases, 219 primary lesions (two largest >= 10 mm per case; 53 bright, 166 dark); HCC-TACE-Seg UID route, 75 cases, 86 lesions (28 bright, 58 dark); same lesion definition and contrast function (`glassbox/audit/contrast.py`). **Every C4KC-KiTS output carries "possibly seen in training (images, via AbdomenCT-1K)".**

## Arms (single object, B = 1, one positive point per lesion, no negative points; point = lesion interior point as v1)
| Arm | label_prompt | point label sent | Role |
|---|---|---|---|
| Kidney zero-shot | 133 | 1 | primary (kidney) |
| HCC supported tumour class | 26 | **3** | primary for HCC (supported tumour class, special point label) |
| HCC zero-shot (matched) | 133 | 1 | primary for the kidney-vs-HCC comparison |
| HCC v1-equivalent | 26 | 1 | secondary, expected to fail the gate |
Special-class check: 129 and 133 are not in `SPECIAL_INDEX` (23, 24, 25, 26, 27, 57, 128), so kidney 133 uses point label 1 as a regular zero-shot class. No kidney tumour class is special. Interior point, coordinate convention (v1's kidney-organ test, Dice 0.90 to 0.955, reused), prediction threshold (probability > 0.5 on the original grid) and annotated-slice restriction are as v1. The runner must send the point label of the arm (v1's runner hard-coded 1).

**Vendored pipeline and MONAI on the zero-shot route** (read 2026-10-16): `vista3d_pipeline.py` accepts any label prompt below 255 when points are given (the supported-label check runs only `if points is None`, lines 245 to 258) and requires a single object with points (line 279); `inferer.py` lines 64 to 77: `if torch.any(class_vector > point_head.last_supported): class_vector = None`, keeping `prompt_class`; MONAI `last_supported=132` (`vista3d.py` line 53), indices above 132 use `zeroshot_embed`, supported ones `supported_embed` (lines 561 to 566).

## Pilot validity gate (before any analysis)
- **Pilot set**, fixed by seed 133 from ground-truth tables only (no model outcome and no v1 result read), stratified, written to [v2_pilot_selection.csv](v2_pilot_selection.csv) before any run (`scripts/w2v2_pilot_select.py`, deterministic: repeated runs give an identical file): per organ 10 lesions, 5 bright + 5 dark, at most one per case; size strata <30 mm: 2 bright + 1 dark, 30 to 60 mm: 2 + 2, >60 mm: 1 + 2 (so >= 3, >= 3, >= 2 per stratum). Run every arm of that organ on its 10 pilot lesions with all predicted masks saved.
- **Per lesion:** passes if predicted/lesion volume < 3x AND the share of predicted voxels inside the lesion > 0.3 (predicted voxels and lesion voxels on annotated slices, as in the validity audit).
- **Per arm:** passes if >= 80% of its pilot lesions pass AND every size stratum present passes >= 60% of its lesions.
- **Fail -> stop, report, no analysis for that arm.** If the kidney arm or the HCC zero-shot arm fails, the kidney-vs-HCC comparison is not run; each arm's primary analysis is run only if its own gate passes. The v1-equivalent arm is expected to fail and is reported for completeness; it is analysed only if it passes.
- Pilot lesions are excluded from the main analysis (lesion-level exclusion; the other lesion of the same case stays). After exclusion the main analysis has 209 kidney lesions (48 bright, 161 dark) and 76 HCC lesions (23 bright, 53 dark). **Stated limit: the pilot uses 3 of the 6 HCC lesions under 30 mm, leaving few small HCC lesions for the main size strata.**

## Primary analysis (valid only for an arm whose pilot passed)
gap = mean coverage in dark lesions minus mean coverage in bright lesions (coverage = share of ground-truth lesion voxels predicted), case-level bootstrap 95% CI (1000 resamples, seed 0), per primary arm. Verdict, in order: **GAP** (CI lower > 0 and point >= 0.10); **NO MEANINGFUL GAP** (CI upper < 0.10); **INCONCLUSIVE** otherwise. **Like-for-like replication = kidney zero-shot and HCC zero-shot (matched) both GAP.** The HCC class 26 + label 3 verdict is reported as the HCC tumour-class result. The kidney-minus-HCC gap difference (matched arms) with CI is descriptive only. (This definition of replication is mine and needs your sign-off.)

## Pre-specified sensitivity (added at approval)
The main analysis (primary gap, Dice gap, diameter-ratio gap, size-adjusted secondary) is also run INCLUDING the pilot lesions (kidney 219, HCC 86 lesions, using the pilot predictions already made with the same code and arm settings). The primary analysis still excludes them. Both versions are reported.

## Key secondary
- **Dice gap** (dark minus bright): Dice between the ground-truth lesion and the prediction on annotated slices (Dice within 10 mm also reported), case-level bootstrap CI.
- **Predicted/GT diameter ratio gap:** largest axial diameter among predicted components overlapping the lesion over the lesion's diameter (v1's measure; the point-component diameter is also stored); gap = mean log2(ratio) in bright minus dark, case-level bootstrap CI.
- Coverage, Dice and RECIST bias (sum of up-to-2 target diameters, as v1) per arm, split by bright and dark.

## Size-adjusted secondary (as v1)
Gap within strata <30, 30 to 60, >60 mm; OLS of coverage on `bright` and log(diameter) with a clustered bootstrap CI. If the adjusted effect disappears (CI includes 0 or opposite sign to the headline), it is stated next to the headline.

## Output and storage
Save ALL predicted masks (full volume, compressed, with CT/ground-truth/prediction spacing and affine and the prompt point and point label) for every arm, pilot and main.

## Named confounds
Possible in-distribution images (C4KC-KiTS via AbdomenCT-1K; tumour labelling there unknown); arterial/corticomedullary-phase KiTS (bright cortex inside the parenchyma shell); point prompts give location; contrast phase and tumour type differ between kidney and HCC; `Mass` may contain cysts; size is related to brightness. Added for v2: the zero-shot route carries no tumour class; v2 was designed after v1's results were seen; the point-label convention was found after v1.

## Mechanism steps
Not part of v2; they would need their own approval and are only considered if a primary shows a GAP.

## Execution rule (approved)
The three PRIMARY arms are kidney 133 / point label 1, HCC 26 / point label 3 and HCC 133 / point label 1. **If all three pass the pilot gate, proceed straight to the main run (all masks saved) with no further sign-off. If any primary arm fails, stop after the pilot and report, with overlays.** The secondary arm (26 / point label 1) failing blocks nothing. The GPU idle times are reported to the researcher.

## Procedure after approval
1. Commit this file and `v2_pilot_selection.csv` (done before any run). 2. The instance is already up; no relaunch. 3. Update the runner (per-arm label and point label, save all masks, pilot mode), unit-test it (including that point label 3 reaches the point head), run the pilot only. 4. Evaluate the gate per arm, report, stop on a failed arm. 5. If gates pass, run the main arms on the non-pilot lesions, then the pre-registered analysis.
