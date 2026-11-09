# W6b smoke tests and scale measurements (training data only; no evaluation image read; 2026-11-09)

Run on one A10 (24 GB), 50 iterations per arm, bundle trainer + `glassbox/seg/w6b_trainer.py`, lambda 1 for the smoke tests. Files: `smoke/`.

| | W6b-FULL | W6b-DEC |
|---|---|---|
| Trainable parameters | 217,964,982 | 21,434,064 |
| Seconds per iteration (median) | 0.787 (W6 BRIGHT: 0.622) | 0.607 (W6 BRIGHT-DEC: 0.422) |
| Peak memory allocated / reserved | 10.9 / 11.2 GB (W6: 9.7 GB) | 7.0 / 7.3 GB (W6: 5.2 GB) |
| 8,000 iterations | about 1.75 h | about 1.35 h |
| Losses finite | yes | yes |
| Mean task loss / mean distillation BCE / KL (50 its) | 0.649 / 0.0020 / 1.7e-05 | 0.578 / 0.0019 / 2.3e-06 |

Checks (first iteration, both arms): the distillation KL is exactly 0.0 and its gradient norm exactly 0.0 (student = teacher); the task loss with the 16 extra distilled prompts equals the task loss without them (FULL 0.33373094 vs 0.33373100; DEC 1.00068212 vs 1.00068212); the teacher's parameter sum is identical before and after the 50 iterations (6262402.1667); the 18-prompt output has shape [18,1,128,128,128] (2 task + 16 distilled), student and teacher distilled logits [16,1,128,128,128]. Unit tests: `tests/test_w6b_distill.py` (5 pass: zero gradient at student = teacher; faint 0.01 haze over an empty teacher class costs about 0.01; loss grows monotonically with disagreement; teacher gets no gradient; sampler), `tests/test_w6b_secondary.py` (1 pass).

Scale measurement (lambda rule, PREREG section 4): BRIGHT's final model against the baseline on 200 training patches: mean KL 8.744e-4 over the 123 classes (liver 1.6e-2, right kidney 4.2e-3, left kidney 9.0e-4, spleen 7.1e-4; median class 2.8e-4); BRIGHT's mean training loss over its last 1,000 iterations 0.5015; resulting lambda_mod 143, lambda_strong 573.

Class presence in 520 training patches (frozen baseline, at least 1,000 voxels at p > 0.5): liver 84%, right kidney 74%, left kidney 34%, spleen 31%; median class 13%; 31 of 123 classes under 5%.
