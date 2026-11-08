# W6 preflight and smoke report (2026-11-08; one A10; reported to the researcher before any arm trains)
Environment (`smoke/env_freeze.txt`, `smoke/env_install.log`): the W4 package set except MONAI: **torch 2.14.1+cu130, MONAI 1.4.0 (the bundle's pin), pytorch-ignite 0.5.1, numpy 2.2.6**, CUDA available on an NVIDIA A10 (23 GB). Weights `vista3d_pretrained_model/model.pt` sha256 `c92bab26d00b4a5d...` (full value in the run reports); bundle `NVIDIA-Medtech/NV-Segment-CTMR` commit `cb921f5c58837c0f42a713855d68b32af88e1cdd`; code hashes `code_hashes.txt` verified on the instance before the runs. Driver used: `scripts/w6_train.py` after the fix of the smoke-mode epoch sizing (commit d332b4a; the first smoke attempt stopped at 8 of 50 iterations because of that bug and is not used).

## 1. Environment parity gate: PASS (`smoke/parity_gate.json`)
The new environment reproduces the W4 baseline class-26 predictions on 3 fixed WAW-TACE series (seeded sample among 396 series whose stored class-26 mask has at least 1,000 voxels): **byte-identical** on 347_2 (8,022 voxels), 360_1 (82,432) and 452_2 (23,591); class 1 (liver) also byte-identical on all three.

## 2. Liver-coverage rule on the baseline (`smoke/liver_rule.json`): tumour-only mapping `[[2,26]]` for ALL arms
Pooled fraction of ground-truth tumour voxels inside the baseline's class-1 mask over the 24 nnU-Net fold-0 validation cases that contain tumour voxels: **0.984** (lowest single case 0.899). The rule (>= 0.50) therefore selects the tumour-only mapping `[[2,26]]` (label set `[0, 26]`): VISTA's liver class includes tumours, and the mapping `[[1,1],[2,26]]` would have taught a new convention. The edit still uses the local labels (liver 1, tumour 2) because the relabelling is the last transform.

## 3. 50-iteration smoke tests (`smoke/BRIGHT/run_report.json`, `smoke/BRIGHT-DEC/run_report.json`; patch 128^3, batch 1, AMP, 8 cached cases, mapping `[[2,26]]`, label set `[0, 26]`)
| | full fine-tune (BRIGHT) | decoder-only (BRIGHT-DEC) |
|---|---|---|
| iterations done | 50 of 50 | 50 of 50 |
| trainable parameters | 217,964,982 (all) | 21,434,064 (9.8%) |
| **peak memory allocated / reserved** | **9.27 GB / 9.73 GB** | **5.17 GB / 5.30 GB** |
| median seconds per iteration (it/s) | 0.60 s (1.66 it/s) | 0.42 s (2.37 it/s) |
| loss finite | yes | yes |
| patches with an edited tumour | 7 of 50 (14%) | 3 of 50 (6%) |
Peak memory is far below the 22 GB limit at 128^3: **the 96^3 fallback is not needed.** (The meta-device estimate of 12 to 19 GB was conservative; mixed precision keeps the activations smaller.) The edit fraction is below 25% because only patches that contain a qualifying tumour component can be edited (tumour-centred crops are about half of the patches with the tumour-only mapping, tumour-free cases have none), and 50 patches is a small sample. Loss values of exactly about 1.0 are patches without tumour: the Dice term of an empty target with the official smoothing sits at about 1.
**Transform order (training; asserted before the first iteration, validation chain in the run report):** `LoadImaged, CropForegroundd, Orientationd, Spacingd, CastToTyped, EnsureTyped, SpatialPadd, SpatialPadd, ClassesToIndicesd, RandCropByLabelClassesd, BrightTumourShiftd, ScaleIntensityRanged, RandZoomd, RandSimulateLowResolutiond, RandGaussianSmoothd, RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised, Relabeld`; validation: `LoadImaged, CropForegroundd, ScaleIntensityRanged, Orientationd, Spacingd, CastToTyped, EnsureTyped, SpatialPadd, Relabeld` (official order).

## 4. Projected hours per arm (estimates from these measurements; the time rule is warn-only)
| part | A10 hours | basis |
|---|---|---|
| bundle cache of 104 cases (once per run) | about 0.2 | estimate (not measured in the smoke: 8 cases) |
| training, 8,000 iterations | **1.3 (full), 0.9 (decoder-only)** | 8,000 x 0.60 s, 8,000 x 0.42 s |
| validation (10 cases about every 1,040 iterations, 8 times) | about 0.25 | estimate |
| WAW-TACE inference, 396 series | **2.4** (to 5.5 if the parity run's 3.4 minutes for 3 series including the model load is representative) | 21.5 s per series in W4 (measured); not re-measured here |
| HCC-TACE-Seg inference (75 cases) | 0.4 | estimate |
| forgetting guard (spleen 41, KiTS 60, 20 all-class CTs, tumour-free livers) | 0.9 | estimate |
| **per full arm** | **about 5.5 (range 5.0 to 8.6)** | |
| per decoder-only arm | about 5.0 | |
Arms 1 to 3 (BRIGHT, SHAM, PLAIN): about 17 GPU-hours plus 0.9 for the baseline guard; with arms 4 and 5: about 28 (single A10), about half the wall time on two. Nothing was trained beyond 50 smoke iterations; the smoke weights were not kept or evaluated.
