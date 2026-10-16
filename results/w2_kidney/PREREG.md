# Pre-registration W2: kidney replication (C4KC-KiTS)

Status: APPROVED with edits (2026-10-16 decisions), committed on `research/roadmap` BEFORE any model run. No model has touched C4KC-KiTS or the HCC point arm. The model is run only after the GPU instance is relaunched, and only as written here. Deviations go to `results/w2_kidney/DEVIATIONS.md`. **Rule from now on: any change to inclusion or exclusion criteria needs the researcher's sign-off before it is applied, even pre-analysis.**

## Question
Given where a tumour is, is VISTA-3D's coverage lower for tumours that are brighter than the surrounding organ parenchyma, in kidney (C4KC-KiTS) as in liver (HCC)?

**What point prompts do and do not measure.** A point prompt gives the model the tumour's location. The headline therefore measures appearance-driven under-segmentation given the location, NOT detection. It cannot show that the model fails to find bright tumours, and it is not comparable to the existing class-26 HCC results (a different analysis, already reported in `results/SUMMARY.md`).

## Data
- C4KC-KiTS (TCIA; the 210 training cases of the KiTS19 challenge), converted to NIfTI on the persistent filesystem of the GPU instance: 210 cases, segments `Kidney` (1) and `Mass` (2). TCIA licence CC BY 3.0. The TCIA page describes phase inconsistently ("corticomedullary" in the description, "arterial phase in every case" in a note); the AbdomenCT-1K paper says the KiTS scans "were acquired on the arterial phase". Phase is treated as arterial/corticomedullary, not verified here.
- `Mass` is whatever the KiTS19 annotators called tumour; cysts are not a separate label here and may be inside `Mass`.
- HCC comparison arm: the HCC-TACE-Seg UID-route cohort, 75 cases / 86 lesions (same lesions as W1).
- **In-distribution flag, carried on every output: all of C4KC-KiTS is "possibly seen in training (images, via AbdomenCT-1K)".** VISTA-3D used AbdomenCT-1K (1,050 volumes), which contains 300 KiTS cases (the 210 C4KC-KiTS training cases plus 90 test cases) with a kidney label. Which cases were in the 64% training split is not public; no case matching is attempted.

## Prompting (class indices verified)
- `cache/bundle_metadata.json` (the `vista3d132` bundle): `"129": "kidney mass"` and `"130": "liver tumor"`. The vendored pipeline (`cache/hf/vista3d_pipeline.py`) removes 16, 18, 129, 130 and 131 from the label-only list ("Undefined label prompt detected. Provide point prompts for zero-shot."), so both 129 and 130 are point-only classes; with points the label prompt must be a single object. 130 is therefore the like-for-like liver counterpart of 129.
- **Primary arm, both organs, identical protocol:** one positive point per lesion, `point_labels = [1]`, no negative points; `label_prompt = [129]` for kidney lesions and `[130]` for HCC lesions. The point is the lesion's interior point: the voxel of maximum 3D distance to the lesion boundary (distance transform with true spacing, ties broken by the lowest voxel index), not the centroid. Points are given in the input image's voxel coordinates, with the convention checked by a unit test (a known point on a synthetic object returns that object) before any scored run.
- **HCC sensitivity arm:** `label_prompt = [26]` (hepatic tumour, label-promptable) with the same point.
- The automatic kidney mask (classes 5 and 14) is descriptive only, not an arm. The existing class-26 HCC results are not re-run.

## Lesions and contrast (ground truth only; identical function to W1)
- Lesion: 26-connected component of `Mass`, longest axial Feret diameter >= 10 mm, the two largest per case (the exp8 rule).
- c = lesion mean HU minus kidney-parenchyma mean HU in the 5-20 mm shell (3D, true spacing), by `glassbox/audit/contrast.py`, parenchyma = label 1, which excludes all `Mass` voxels by construction; fallback to all non-tumour kidney voxels if the shell has fewer than 200 voxels. Bright = c > 0 HU. Phantom tests: `tests/test_w2_kidney_contrast.py`.
- **Ground-truth feasibility, computed 2026-10-16 before this file was committed and before any model run** ([gt_summary.json](gt_summary.json), [gt_lesions.csv](gt_lesions.csv)): 210 cases, 219 primary lesions, 53 bright (24.2% [18.8, 30.0]), 166 dark. The HCC arm has 28 bright of 86 lesions (W1), below the roughly 40 bright lesions a brightness test wants: its gap CI will be wide and the rule below applies unchanged.
- **Inclusion:** every lesion in the ground-truth table enters; a lesion the model does not segment counts as coverage 0; no lesion is dropped after a model run. Any change to this needs sign-off.

## Outcomes (per arm, kidney and HCC separately)
- Per lesion: **coverage = share of ground-truth lesion voxels that are predicted** (headline); lesion Dice (Dice between the lesion and the predicted voxels within 10 mm of it, as in the HCC counterfactual); detected = coverage >= 0.10 (reported; expected near ceiling because the point gives location).
- Per case: RECIST bias as in `scripts/recist_shift.py` (sum of the up-to-2 target diameters; predicted diameter = largest among overlapping predicted components, a miss counts as 0 mm; relative error of the sum versus ground truth).
- Coverage, Dice and RECIST bias are all reported per arm, split by bright and dark.

## Decision rule (headline)
gap = mean coverage in dark lesions minus mean coverage in bright lesions (positive = brighter tumours are covered less), computed separately for kidney and for HCC under the identical point protocol, with a case-level (patient-clustered) bootstrap 95% CI (1000 resamples, seed 0, percentile). Verdict per tumour type, mutually exclusive, applied in this order:
1. **GAP:** CI lower bound > 0 and point estimate >= 0.10.
2. **NO MEANINGFUL GAP:** CI upper bound < 0.10.
3. **INCONCLUSIVE:** otherwise.
**Replication = kidney and HCC are both GAP.** The kidney-minus-HCC difference in the gap, with its CI, is reported and is descriptive only.

## Size-adjusted secondary (pre-specified)
Brightness and size are related (bright and dark tumours differ in size), and coverage depends on size. Reported next to the headline, never instead of it:
- the gap within each size stratum (<30, 30-60, >60 mm), with CIs;
- a regression of lesion coverage on `bright` and `log(diameter)` (ordinary least squares with an intercept), with the coefficient on `bright` and a patient-clustered bootstrap 95% CI (1000 resamples, seed 0).
**If the adjusted `bright` coefficient's CI includes 0 or has the opposite sign to the headline gap, that is stated next to the headline.**

## Mechanism steps (only if the headline shows a GAP; each needs its own approval)
Encoder probe (with random-init and raw-HU controls), LEACE erasure and patching, specified exactly as in HCC (`results/voxel_probe/`, `results/mech/PREREG.md`); the encoder is class-agnostic, so these are prompt-independent. Nothing here is run before approval.

## Named confounds (stated in advance, repeated in the write-up)
- **Possible in-distribution images:** C4KC-KiTS may have been seen in training through AbdomenCT-1K, and whether KiTS tumours were labelled as kidney there is unknown (VISTA-3D may have been trained to call these tumours kidney).
- **Arterial-phase KiTS:** scans are arterial/corticomedullary, so bright renal cortex lies inside the parenchyma shell and the parenchyma mean is a coarse reference (cortex and medulla differ strongly; the hilum is not separately labelled).
- **Point prompts give location:** the headline is appearance-driven under-segmentation given location, not detection.
- Also: contrast phase and tumour type differ between kidney and HCC; `Mass` may contain cysts; size is related to brightness (hence the size-adjusted secondary).

## Procedure
1. Commit this file. 2. Relaunch the instance (after the researcher confirms). 3. Unit-test the point coordinate convention; run the primary arms (kidney 129; HCC 130) and the HCC class-26 sensitivity arm. 4. Compute outcomes, gaps, size-adjusted secondary. 5. Report at Gate 3 and stop.
