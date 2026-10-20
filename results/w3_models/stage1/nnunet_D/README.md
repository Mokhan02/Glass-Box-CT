# Model D (nnU-Net v2, Task03, fold 0) training finished; sanity gate PASSED

Seeded trainer (`nnUNetTrainerW3Seed`, seed 20261020), 1000 epochs, one uninterrupted attempt (no restarts), started 2026-10-19 22:19 America/New_York, last epoch 2026-10-20 15:46, final validation done 16:29 (all times America/New_York). `checkpoint_final.pth` sha256 prefix `4c79123fdf454c77` (instance persistent filesystem, not committed). torch.compile training is not bitwise deterministic (PREREG_stage1).

Sanity gate (PREREG_stage1: stop if fold-0 validation tumour Dice < 0.537 or liver < 0.93), nnU-Net's own validation of the 27 held-out Task03 cases (`fold0_validation_summary.json`, full-resolution, mirroring on):
| class | fold-0 mean Dice | gate | published 5-fold CV (arXiv 1904.08128) |
|---|---|---|---|
| liver | **0.9651** (27 cases, per-case min 0.920) | >= 0.93: pass | 0.9571 |
| tumour | **0.6466** (nnU-Net's mean over the 26 cases that have a tumour; 1 validation case has none and gives NaN) | >= 0.537: pass | 0.6372 |

No HCC inference with model D has been run.
