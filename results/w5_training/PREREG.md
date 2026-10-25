# W5 PREREG (APPROVED by the researcher with one edit, the ranked optional arms): does training with bright-lesion contrast augmentation close the brightness gap in the self-trained nnU-Net?

Branch `research/roadmap`. Written before any W5 training run and before any W5 result. Same rules as W3: the researcher signs off before any GPU run, code is committed and hashed before runs, commits list files explicitly, deviations go to `results/w3_models/DEVIATIONS.md`, and `RESEARCH_LOG.md` stays append-only. Infrastructure failures (OOM, SSH, preemption) are resumed without asking only if no code or config changes and no results were read, each logged as a deviation.

## Question and prediction (stated before any W5 result)
W3 found a bright-versus-dark gap in model D (nnU-Net, fold 0, Task03 only, whose training tumours are 1.1% bright by the W1 audit). W5 asks whether the gap is caused by the training data's contrast distribution: retrain with an augmentation that shows the model bright tumours and see whether the gap closes.

**Prediction: BRIGHT-50 closes the bright-versus-dark gap compared with both D0 and SHAM-50.** If it does not (CI does not exclude a gap reduction smaller than 10 points), that counts against the simple version of the training-data explanation for this model and recipe (it would not by itself show that data play no role: the augmentation is a crude stand-in for real bright lesions, and this is one seed per arm).

## Arms
| arm | trainer class | augmentation probability p | tumour-minus-local-liver shift range (HU) |
|---|---|---|---|
| **D0** | existing model D (`nnUNetTrainerW3Seed`, no new run) | none | none |
| **BRIGHT-50** | `nnUNetTrainerW5Bright50` | 0.5 | U(+10, +60) |
| **SHAM-50** | `nnUNetTrainerW5Sham50` | 0.5 | U(-60, -10) |
| **D0-seed2** (OPTIONAL, rank 1) | `nnUNetTrainerW5D0Seed2` | none (model D's recipe, no augmentation) | none; **seed 20261121** (fixed now; D0 is seed 20261020) |
| **BRIGHT-25** (OPTIONAL, rank 2) | `nnUNetTrainerW5Bright25` | 0.25 | U(+10, +60) |

**Optional arms are ranked: (1) D0-seed2, (2) BRIGHT-25. Which of them run is decided at the GPU sign-off by the number of GPUs available, in that rank order, before any W5 training starts; it is never decided on any W5 result.** Nothing in the verdict depends on either.
- **D0-seed2** (the same recipe as model D with a different fixed seed) is a training-variance estimate only: next to every BRIGHT-50 contrast, report **|gap(D0) minus gap(D0-seed2)|** for detection and for Dice, descriptive, **not part of the verdict**. Its seed is fixed now (20261121) so it cannot be chosen after seeing a result.
- **BRIGHT-25** exists only for the dose trend (secondary).

SHAM-50 is the procedure control: the same transform, the same probability and the same magnitude of shift (10 to 60 HU relative to the liver), but the result stays below the liver. It shows what the augmentation procedure itself (extra intensity edits to tumours, a changed random stream) does without teaching "bright tumour".

## Augmentation (online, inside a custom nnU-Net v2 trainer transform acting on image and segmentation)
Code: `scripts/w5_nnunet_trainer.py` (core function `shift_tumour_contrast`, transform `TumourContrastTransform`, three trainer classes). For each training sample (patch), with probability p, and independently for each tumour connected component in the patch:
1. Work in HU: nnU-Net's preprocessing has already clipped CT to [-15, 197] HU (the 0.5 and 99.5 percentiles of the training foreground) and z-scored it with mean 99.48 and std 37.14; the transform inverts that, edits, clips to the same window and re-normalises. **The model therefore sees the same window in training as at inference** (and a bright lesion cannot exceed 197 HU in either). The constants are asserted against the plan at trainer start.
2. **Local liver mean** = mean HU of liver-labelled voxels within 12 voxels (about 9 to 12 mm) of the component, excluding every tumour voxel; if fewer than 200 such voxels, the liver voxels of the whole patch; if still fewer than 200, the component is left unchanged.
3. Draw delta from the arm's range and set the target tumour mean = local liver mean + delta.
4. **Additive offset (texture preserved), feathered:** a Gaussian-smoothed mask (sigma 1 voxel) weights a constant offset, scaled so that the mean over the component of the edited image equals the target exactly (before clipping). Voxels outside the lesion are changed only in the feathered rim (unit tests: unchanged beyond about 8 voxels from the lesion).
5. Clip to the CT window. Components under 30 voxels are left unchanged. **The segmentation is never modified.**
6. The transform runs first in the chain, before nnU-Net's own spatial and intensity augmentations, so those transform image and labels together.

**The ranges (+10 to +60 HU for BRIGHT, the mirrored -60 to -10 for SHAM) were fixed by the researcher from physiology (the size of lesion-to-liver attenuation differences that make a lesion visibly brighter or darker than the liver), NOT from any HCC-TACE-Seg measurement, and are not tuned.** No HCC data were used to choose p, the ranges, the shell or the feathering.

Unit tests (`tests/test_w5_aug.py`, synthetic volume): the tumour mean lands at the target and on the intended side of the liver (15 volumes per arm); SHAM stays dark; the segmentation is bit-identical; p = 0 returns the input unchanged; interior texture is preserved (offset only) and the change is local; output is inside the window; components get independent deltas; small or liver-less components are untouched; the application rate matches p. An integration check on the instance (CPU only, no GPU) ran the whole nnU-Net v2.8.1 training transform chain with the new transform on a real preprocessed Task03 patch (shapes, dtypes and label equality fine; a tumour at 87.8 HU became 150.4 HU against a liver mean of 121.4).

## Training: identical to model D except for the transform
nnU-Net v2.8.1, `3d_fullres`, plan `nnUNetPlans` (Dataset003_Liver, spacing 1.0 x 0.768 x 0.768 mm, patch 128^3, batch 2), fold 0 with nnU-Net's own `splits_final.json`, **1000 epochs x 250 iterations, seed 20261020** (Python, NumPy, torch and CUDA seeded before construction, as in `nnUNetTrainerW3Seed`), same GPU type (A10), same torch and nnU-Net versions. The trainer subclasses `nnUNetTrainer` as `nnUNetTrainerW3Seed` does and changes only `get_training_transforms` (prepending the transform). **Limits:** training is not bitwise deterministic (torch.compile, cuDNN autotuning, multi-process data workers), the new transform consumes additional random numbers, so each arm's random stream differs from D0's even at the same seed; each arm is one run, so training-run-to-training-run variance is not measured (unless D0-seed2 is run, which gives one descriptive estimate from two runs), and **the patient bootstrap does not cover it**. Resume after an infrastructure failure continues from a re-seeded state (as in W3).

Pre-run checks (CPU, before the sign-off for GPU): (a) tests above pass; (b) an application audit that runs the dataloader on a few hundred samples per arm and reports the share of samples edited, the delta distribution and the share of components skipped (committed with its output before any run); (c) the trainer file's hash is recorded in `results/w5_training/code_hashes.txt`. **A 5-epoch GPU smoke/benchmark of each trainer (confirming epoch time is not slower than D's 61 s and that no errors occur) needs its own sign-off.**

## Sanity gate (per arm, on the finished fold-0 validation; stop and report on failure; no HCC inference before it is reported)
**Task03 fold-0 validation liver Dice >= 0.93 and tumour Dice >= 0.597** (D's 0.647 minus 0.05; D's liver Dice was 0.965). Computed from nnU-Net's own `validation/summary.json` ("mean" of labels 1 and 2). An arm that fails is reported and excluded from the verdicts.

## Primary analysis (HCC cohort 74 cases / 85 lesions, 28 bright / 57 dark; W3 Stage 1 code unchanged)
Each arm is run on the HCC cohort exactly as D was (`nnUNetv2_predict` CLI defaults: mirroring on, step 0.5, no post-processing; the alignment gate and quality flags of the W3 PREREG applied per arm, HCC_019 scored against its corrected label, HCC_027 excluded from the primary and run as the sensitivity), and analysed with the committed `scripts/w3_analyze.py` with no change.

- **Per-arm gaps:** detection gap (dark minus bright detection rate) and Dice gap (dark minus bright mean Dice), case-level bootstrap, 1000 resamples, per-arm seed (SHA-256 of the arm name, as in W3).
- **Contrasts (the primary W5 outcome):** for each metric, **Delta = gap(BRIGHT-50) minus gap(comparator)** for comparator = D0 and comparator = SHAM-50 (negative = BRIGHT-50 narrows the gap). Paired patient-level bootstrap: the same resampled cases for both arms in each replicate; 95% percentile CI.
- **Verdict per metric and comparator (margin 0.10, structured like W3):** **CLOSES** if the CI upper bound < 0 and the point estimate <= -0.10 (for detection, 10 percentage points); **DOES NOT CLOSE** if the CI lower bound > -0.10; **INCONCLUSIVE** otherwise, in that order.
- **Overall label for the prediction:** "CONFIRMED (detection and Dice)" if both metrics CLOSE against both comparators; "CONFIRMED (detection only)" or "(Dice only)" if only one metric does so against both; "NOT CONFIRMED" if either comparison is DOES NOT CLOSE on both metrics; otherwise "INCONCLUSIVE". The prediction (verdict) is read from the two contrasts, never from one arm alone; D0 versus SHAM-50 is reported as the procedure effect.

## Secondary (reported with the primary, never instead of it)
- Sum-of-diameters bias (median signed error, same definition as the W3 RECIST analysis) and overall tumour Dice per arm, with the bright and dark breakdown.
- **Training-variance reference (only if D0-seed2 is run):** |gap(D0) minus gap(D0-seed2)| for the detection gap and the Dice gap, printed beside each BRIGHT-50 contrast, with D0-seed2's own gaps and sanity gate; descriptive, no verdict, no pooling of D0 and D0-seed2.
- **Dose trend (only if BRIGHT-25 is run):** detection and Dice gaps at p = 0 (D0), 0.25, 0.5, reported with bootstrap CIs and a linear trend in p; descriptive, no verdict.
- **Mechanism on BRIGHT-50 (the committed M1 and M2, unchanged, same hook `encoder.stages[3]`, same cohort and controls):** does the concept stop being brightness-dependent? Reported as M1's own categories for BRIGHT-50 (D0 was LOST LATE) and M2's own verdict, side by side with D0. No new rule; a change of category (for example LOST LATE to REPRESENTED) is reported as a change, not as proof of mechanism.
- SHAM-50 M1 and M2 only if BRIGHT-50 differs from D0 in either, to see whether the procedure alone explains it.

## Limits and named confounds
One seed per arm and one fold (training variance is unmeasured); a single global p and range per arm; synthetic brightening of Task03 tumours (which are mostly dark and heterogeneous) is not the same as real hypervascular lesions, so a null does not exclude the data explanation; the CT window clip at 197 HU limits how bright an edit can look; the local liver mean uses the model's training labels and a 12-voxel shell; the HCC cohort mixes acquisitions and its contrast phase cannot be labelled; D0 was trained with a different code path (no extra transform), so BRIGHT-50 versus D0 includes procedure effects, which is why SHAM-50 is the primary control; the 28 bright lesions make gap CIs wide, as in W3.

## Run estimate (not a commitment; nothing runs without the researcher's sign-off)
Measured Gate 0 rate: 61.1 s/epoch on one A10 (the transform adds CPU work only: 0.19 s for the whole chain on one patch on this instance; to be confirmed by the smoke run), so about **17.0 h per arm (about 17.3 h with warm-up, validation and checkpointing)**.
- **Required (BRIGHT-50 + SHAM-50):** 2 x 17.3 = **34.6 GPU-h; about 35 h wall on one A10 sequentially, or about 17.5 to 18 h wall on two GPUs in parallel**.
- **Optional, in rank order:** D0-seed2 +17.3 GPU-h (about 52 GPU-h total, or parallel on three GPUs about 17.5 to 18 h wall); then BRIGHT-25 +17.3 GPU-h (about 69 GPU-h total, or parallel on four GPUs about 17.5 to 18 h wall). With two GPUs only the two required arms run.
- **After training, per arm:** gate (CPU/inference minutes), HCC inference about 25 min, W3 analysis on CPU; M1 and M2 on BRIGHT-50 (about 5 GPU-h in the W3 Stage 2 runs). Total GPU-hours: about 41 required, about 58 with D0-seed2, about 75 with both optional arms.
- Checkpoints every 50 epochs on the persistent filesystem and automatic resume, as for D (`scripts/w3_train_d.sh` pattern).

## Amendment 1 (2026-10-25, before any W5 training started; the researcher's decision at the GPU sign-off): the optional arms are not run
Two GPU instances are available, so **only the two required arms run: BRIGHT-50 (instance 1) and SHAM-50 (instance 2, started once its 5-epoch smoke has passed). By the ranked rule above, D0-seed2 and BRIGHT-25 are NOT run. This was decided by GPU count at the sign-off, before any W5 training started, and not on any W5 result.** Consequences, stated now: there is **no training-variance estimate** (|gap(D0) minus gap(D0-seed2)| cannot be reported, and BRIGHT-50 contrasts carry no second-seed reference), and **no dose trend** (the BRIGHT-25 secondary is dropped). **The single-seed training-variance limitation stands as already stated in "Limits and named confounds"; the patient bootstrap does not cover it, and every W5 reading is labelled single-seed.** The verdict rule is unaffected (neither optional arm entered it). The trainer classes for the optional arms stay committed and unused.
