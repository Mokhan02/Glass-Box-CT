# W6b PREREG (approved with changes by the researcher 2026-11-09; amended below; the real runs wait for the researcher's go): bright-tumour fine-tuning with self-distillation to the frozen baseline

Research use only; no clinical claims. Written 2026-11-09; amended the same day after the researcher's review and after the training-data-only smoke tests (no evaluation image read, no real run). Derivatives of NV-Segment-CT stay internal (licence). Every number marked ESTIMATE is derived from W6 runs and must be confirmed in the smoke tests below.

## 1. Why

W6 showed that the bright-tumour edit closes the arterial gap (BRIGHT: bright arterial detection 77.3% against 21.5% baseline, replicated by seed 2 at 76.4%), but every W6 arm, including the unedited PLAIN arm, fails the forgetting guard (kidney Dice -0.030 to -0.056; 20 to 32 of 48 classes above the 10 mL floor below the 0.95 drift Dice). The forgetting therefore comes from tumour-only fine-tuning (all non-tumour classes unsupervised), not from the edit. W6b asks whether keeping the network's non-tumour behaviour pinned to the frozen baseline removes the forgetting without losing the bright-tumour gain.

## 2. Arms (amended 2026-11-09 after the researcher's review; lambda set by the fixed rule of section 4)

| Order | Arm | Trainable | lambda | Where |
|---|---|---|---|---|
| 1 | W6b-FULL-L143 (lambda_mod) | whole network (as BRIGHT) | 143 | instance 1, now |
| 2 | W6b-FULL-L573 (lambda_strong) | whole network (as BRIGHT) | 573 | instance 2, as soon as TotalSegmentator finishes |
| 3 (last) | W6b-DEC-L14343 | `image_encoder.up_layers_auto` + `class_head` (as BRIGHT-DEC) | 143 | wherever a GPU frees first |

Both full-network lambda values are pre-registered, both are run and BOTH are reported; there is no selection between them (no "best lambda"). Each arm is judged on its own against the primary rule of section 6. Go given by the researcher on 2026-11-09 (no further sign-off needed); same rules as W6: frozen evaluation, and no arm's WAW-TACE results are opened until both full-network arms are scored; the report lists the primary for each arm first.

Unchanged from W6 (same code, hashes of the shared parts, seeds recorded): data (104 training cases, 128^3 patches, tumour-only mapping [[2,26]], label set [0,26]), bright edit (p = 0.25, +10 to +60 HU, shell 8 voxels, feather sigma 1, applied on raw HU before the intensity window), AdamW LR 5e-5 constant, 8,000 iterations, final checkpoint only, class-26 inference prompting, WAW-TACE scoring, HCC-TACE-Seg pass, forgetting guard (frozen analysis, streaming drift script), W4b/W6 analysis scripts. Seed 20261110 for every arm (as BRIGHT and BRIGHT-DEC); single run per arm unless a second seed is approved (section 9).

## 3. Distillation term (the only new training component) AMENDED: voxelwise BCE against the teacher's soft probabilities

- Teacher: the frozen baseline weights (the same file the arms start from), eval mode, no gradient, same augmented patch as the student (edited patch included).
- Per iteration, K = 16 classes are drawn uniformly without replacement from the 123 classes of the guard's list (every class index of the bundle's label dictionary except background 0 and the tumour class 26; indices 1 to 132 with 26 removed; kidney is classes 5 and 14, spleen class 3, liver class 1). The list only, no evaluation image. The K classes are scored in the SAME student forward as additional class prompts (the class head scores each prompt independently; checked in the smoke test), and the teacher is run once for them.
- L_distill = mean over the K classes and all patch voxels of BCE(student logit, teacher probability) with teacher probability = sigmoid(teacher logit). The teacher is a constant. At student = teacher the GRADIENT is exactly zero (d/dlogit = p_student - p_teacher) while the value equals the teacher's mean binary entropy; the entropy-free value (the KL divergence, same gradient) is logged as a monitor.
- Why BCE and not the soft-Dice of the first draft (researcher's decision): a Dice term is unstable when a class is almost empty in the patch (a faint student haze reads as a total mismatch), and BCE against soft targets is the standard distillation choice. Unit tests (all pass, `tests/test_w6b_distill.py`): (a) student = teacher gives zero gradient and zero KL; (b) an empty-teacher class with a student haze of 0.01 gives a loss of about 0.01, not about 1; (c) the loss grows smoothly and monotonically with disagreement in both directions; plus the class sampler (distinct, reproducible, every class reachable, never 0 or 26).
- Total loss: L = L_task + lambda x L_distill, L_task the unchanged W6 DiceCE (smooth_dr 0.01, smooth_nr 1e-4, the bundle's own trainer code). The tumour class 26 and background are not distilled.
- SCALE NOTE: lambda keeps its meaning as the weight on this term relative to the task loss, but the scale of the term differs from the Dice draft. A mean over all voxels is small (most voxels are background for nearly every class): an absent class costs about 0.01 per voxel for a 0.01 haze, and the KL of a faintly different student is far below 1. For that reason lambda is set by the scale rule of section 4 (143 and 573), not 1 and 5 of the first draft. Smoke-test magnitudes: mean BCE 0.002 / KL 1.7e-05 (full network), 2.3e-06 (decoder-only) over 50 iterations against a task loss of about 0.6.
- Requires the custom trainer `glassbox/seg/w6b_trainer.py` (subclass of the bundle's trainer; the task part of an iteration is the bundle's code unchanged) and driver `scripts/w6b_train.py`, both hashed before any run.

## 4. Distillation weight lambda: set by a rule fixed before the arms run (training data only, no WAW-TACE)

Rule (researcher, 2026-11-09): measure BRIGHT's divergence from the baseline on 200 training patches, and choose lambda_mod and lambda_strong so that lambda x (BRIGHT-level distillation term) equals 0.25 x and 1.0 x the mean task loss. If lambda = 1 and 5 already land within 2x of those targets, keep 1 and 5.
Definitions fixed now: "BRIGHT-level distillation term" = the mean over all 123 distillable classes and all voxels of the KL divergence (the BCE without the teacher's entropy, which has no gradient) between BRIGHT's final W6 model and the baseline on 200 training patches drawn by the arm's own training pipeline (expected value of the term under uniform class sampling); "mean task loss" = BRIGHT's own mean training loss over the last 1,000 of its 8,000 W6 iterations (training log).
Measured (files in `smoke/`): mean KL = 8.744e-4 (liver 1.6e-2, right kidney 4.2e-3, left kidney 9.0e-4, spleen 7.1e-4, median class 2.8e-4); mean task loss = 0.5015. Targets: 0.25 x 0.5015 = 0.1254 and 1.0 x 0.5015 = 0.5015.
Result: lambda_mod = 0.1254 / 8.744e-4 = 143.4, lambda_strong = 0.5015 / 8.744e-4 = 573.5. The 1 and 5 of the first draft are 143x and 115x below these targets, so the 2x clause does not apply and the rule gives **lambda_mod = 143 and lambda_strong = 573** (rounded to integers). Reading: a student that drifted as far as BRIGHT did would pay a distillation term worth 25% (moderate) or 100% (strong) of the task loss; because the leash makes the student drift less, the term actually paid will be smaller than that. Larger or per-class-specific weights, and any other value, are not run unless the researcher approves a new amendment. The scale note of section 3 stands (lambda 143 or 573 on a BCE-type term whose per-voxel values are small).

## 5. Evaluation (frozen code, unchanged)

Identical to W6: WAW-TACE inference and scoring of both halves; HCC-TACE-Seg; forgetting guard (spleen, kidneys, liver, kidney-tumour side check, false positives, drift of the 123-class list at the 10 mL floor); W4b/W6 analysis (`scripts/w6_analyze.py`), streaming drift script. Hash files verified before each step as in W6.

## 6. Primary endpoint (STRICT, unchanged; each arm is judged separately; no combined claim, no selection between arms)

An arm SUCCEEDS only if BOTH hold:
1. Non-inferiority on bright arterial detection against BRIGHT (seed 1): on the 242 arterial-bright tumours (same set as W6), the paired difference (W6b arm minus BRIGHT) in detection has a patient-clustered bootstrap 95% CI (B = 2,000) whose lower bound is above -0.10. (Reference noise: BRIGHT against BRIGHT seed 2 differ by +0.008, 95% CI [-0.017, +0.034]; BRIGHT against BRIGHT-DEC by +0.095 [+0.056, +0.137].)
2. It passes the forgetting guard as defined in W6: kidney Dice drop at most 0.02 (mean) with the CI lower bound above -0.03, spleen and liver likewise, false positives within 1.25 x baseline median + 0.5 mL, and drift Dice at least 0.95 for every class above the 10 mL floor.

Outcomes are reported as: SUCCEEDS; GUARD ONLY (guard passes, detection non-inferiority fails); DETECTION ONLY (detection holds, guard fails: the "fix with forgetting cost" tag stays); NEITHER.
Stated in advance: the full 0.95 drift criterion on all 48 classes is strict (BRIGHT fails 32, the best W6 arm, DEC, fails 20; the lowest class mean Dice above the floor was 0.75 to 0.81). A partial reduction is plausible and a full pass is uncertain; the graded secondaries below make partial progress visible.

## 7. Secondaries (descriptive, no pass/fail beyond section 6)

GRADED, explicit (script `scripts/w6b_secondary.py`, written, tested and hashed before the first run; 95% bootstrap CIs, B = 2,000, series-level for classes, case-level for kidney, resamples shared between arms):
- Failing-class count against BRIGHT: for each arm, the number of the 48 above-floor drift classes with mean Dice below 0.95, with its CI, and the paired difference to BRIGHT's 32 with its CI.
- Kidney Dice change against BRIGHT: for each arm, the paired change in kidney Dice (60 fixed cases) arm minus BRIGHT with its CI, and the arm's change against the baseline with its CI (BRIGHT: -0.056 [-0.065, -0.048]).

Other descriptive items: lowest class mean Dice above the floor, spleen, liver, side check and false positives against BASELINE and against BRIGHT, DEC, PLAIN; detection by contrast group on arterial and portal series, W4b theta and label, HCC-TACE-Seg detection and Dice; over-segmentation tables (no-overlap components, FP volume bands); the matched false-alarm table (exploratory); a paired comparison of each W6b arm with its W6 counterpart.

## 8. Cost estimate (smoke-test measurements for training; other items ESTIMATE from W6 timings, A10 24 GB)

| Item | W6b-FULL (each of lambda 143 and 573) | W6b-DEC-L143 |
|---|---|---|
| Training, 8,000 iterations (smoke test, FULL: 0.787 s/it, peak 10.9 GB allocated; W6 BRIGHT: 0.622 s/it, 9.7 GB) | about 1.75 h | about 1.3 h (ESTIMATE; DEC smoke figure in the smoke report) |
| WAW-TACE inference + scoring (W6: about 2.7 h) | 2.7 h | 2.7 h |
| HCC-TACE-Seg pass | 0.2 h | 0.2 h |
| Forgetting-guard inference (W6: about 1.9 h GPU; the drift and false-positive analysis is CPU, about 2 to 3 h wall, on any instance) | 1.9 h | 1.9 h |
| Total GPU | about 6.6 h | about 6.1 h |

Three arms: about 19 GPU-h; about 7 h wall for the two full-network arms in parallel on two A10 instances, the decoder-only arm after them (about 6 h more) only if time allows. A second seed or another lambda would each add about 6.5 GPU-h and need approval. Estimates include no queue idle time.

## 9. Stated risks and checks before any real run (the real runs need the researcher's go)

Done before the real runs (reported separately in `SMOKE_REPORT.md`): unit tests of the loss and sampler pass; smoke tests on training data only (50 iterations per arm): losses finite; first-iteration distillation gradient exactly 0 and KL exactly 0; the task loss is the same with and without the extra prompts (difference below 1e-6); teacher parameters unchanged; K-class logits have the expected shape; measured s/it and peak memory.

Stated risks:
- Scale (measured, section 4): over the 50-iteration smoke test the term was tiny (mean BCE 0.002, KL 1.7e-05 for the full network) because the student had barely moved; at BRIGHT's level of drift the mean KL is 8.7e-4, which is why lambda is 143 and 573 and not 1 and 5. The liver class carries about 18 times the mean KL, so a large lambda holds the liver prediction hardest.
- Class presence in the training patches (from the frozen baseline's masks on 520 training patches, a class counted present with at least 1,000 voxels at p > 0.5; `smoke/class_freq_baseline_520.json`): liver 84%, right kidney 74%, left kidney 34%, spleen 31%. The median class is present in 13% of patches and 31 of the 123 classes are present in under 5% (25 classes in at least 50%, 67 between 5% and 50%). Kidney and spleen are therefore not rare in the patches, but many other classes are, and uniform sampling over the 123 classes (16 per iteration, 13% per class) means each class is both sampled and present in about 1,040 x its share of the 8,000 iterations (right kidney about 770, left kidney about 360, spleen about 320, a class at 5% about 50). Rarely present classes get little protection. No recipe change.
- The trainer is a subclass of the bundle's trainer with the task code copied; any change in a future bundle version would break the equality check (the smoke test checked equality of the task loss).

The real runs have the researcher's go (section 2). No evaluation image is opened for design; lambda, K and the loss form are fixed here.

## 10. Rules carried over

Pre-register before analysing; report negative results; deviations logged with timing (DEVIATIONS file); never overwrite a running script in place (write a new file, switch at the next arm); no IPs in committed files; no change to the frozen analysis; instances are never terminated by the assistant. A null or mixed result is reported as such, and a guard failure keeps the "fix with forgetting cost" tag.
