# W6b PREREG (DRAFT, NOT SIGNED OFF, NO GPU USE UNTIL THE RESEARCHER APPROVES): bright-tumour fine-tuning with self-distillation to the frozen baseline

Research use only; no clinical claims. Written 2026-11-09, before any W6b code runs. Derivatives of NV-Segment-CT stay internal (licence). Draft status: every number marked ESTIMATE is derived from W6 runs and must be confirmed in the smoke tests below.

## 1. Why

W6 showed that the bright-tumour edit closes the arterial gap (BRIGHT: bright arterial detection 77.3% against 21.5% baseline, replicated by seed 2 at 76.4%), but every W6 arm, including the unedited PLAIN arm, fails the forgetting guard (kidney Dice -0.030 to -0.056; 20 to 32 of 48 classes above the 10 mL floor below the 0.95 drift Dice). The forgetting therefore comes from tumour-only fine-tuning (all non-tumour classes unsupervised), not from the edit. W6b asks whether keeping the network's non-tumour behaviour pinned to the frozen baseline removes the forgetting without losing the bright-tumour gain.

## 2. Arms (two variants, everything else as W6)

| Arm | Trainable | Loss |
|---|---|---|
| W6b-FULL | whole network (as BRIGHT) | task loss + lambda x distillation |
| W6b-DEC | `image_encoder.up_layers_auto` + `class_head` (as BRIGHT-DEC) | task loss + lambda x distillation |

Unchanged from W6 (same code, hashes of the shared parts, seeds recorded): data (104 training cases, 128^3 patches, tumour-only mapping [[2,26]], label set [0,26]), bright edit (p = 0.25, +10 to +60 HU, shell 8 voxels, feather sigma 1, applied on raw HU before the intensity window), AdamW LR 5e-5 constant, 8,000 iterations, final checkpoint only, class-26 inference prompting, WAW-TACE scoring, HCC-TACE-Seg pass, forgetting guard (frozen analysis, streaming drift script), W4b/W6 analysis scripts. Seed 20261110 (as BRIGHT and BRIGHT-DEC); single run per variant unless a second seed is approved (section 9).

## 3. Distillation term (the only new training component)

- Teacher: the frozen baseline weights (the same file the arms start from), eval mode, no gradient, same augmented patch as the student (edited patch included).
- Per iteration, K = 16 classes are drawn uniformly without replacement from the 121 non-background, non-tumour classes of the baseline's class list (the guard's 123-class list minus classes 0 and 26; the list only, no evaluation images). The student and the teacher are both prompted with these K classes on the patch.
- L_distill = mean over the K classes of [ 1 - (2 sum(p_s p_t) + e) / (sum(p_s) + sum(p_t) + e) ], p_s and p_t the student and teacher sigmoid probabilities over the patch, e = 1e-5 (a soft-Dice distance to the teacher's soft prediction; it matches the Dice-based drift criterion; a class absent in both gives 0). Alternative considered: Bernoulli KL per voxel; rejected as primary because its voxel mean is tiny (most voxels are background) and gives a weak gradient.
- Total loss: L = L_task + lambda x L_distill, with L_task the unchanged W6 DiceCE (smooth_dr 0.01, smooth_nr 1e-4).
- The tumour class 26 and background are NOT distilled; the task loss alone governs them.
- Requires a custom training step (teacher forward, class subset prompting); new file `scripts/w6b_train.py` plus a test, hashed before any run. Risk: the bundle's trainer must be extended; this is the main implementation uncertainty and is covered by the smoke tests.

## 4. Distillation weight lambda = 1.0 (proposal) and its justification

- Scale: both L_task (Dice + CE on a two-class problem) and L_distill (a Dice distance) live on [0, 1] per term, so lambda = 1 gives the two the same nominal weight.
- The distillation gradient is exactly zero at iteration 0 (student = teacher) and grows only as the student drifts: the term is self-limiting and starts weaker than lambda suggests. A larger lambda mainly tightens the leash late in training.
- No tuning on WAW-TACE or any evaluation series: lambda is fixed here, before any W6b run. If the primary fails at lambda = 1, no lambda search is run; a new value (candidate 5) would be a new, separately pre-registered amendment with the researcher's sign-off, reported as a second experiment.
- Honest uncertainty: nothing in the W6 data says how large lambda must be; 1.0 is a neutral first value, not a calibrated one.

## 5. Evaluation (frozen code, unchanged)

Identical to W6: WAW-TACE inference and scoring of both halves; HCC-TACE-Seg; forgetting guard (spleen, kidneys, liver, kidney-tumour side check, false positives, drift of the 123-class list at the 10 mL floor); W4b/W6 analysis (`scripts/w6_analyze.py`), streaming drift script. Hash files verified before each step as in W6.

## 6. Primary endpoint (each variant is judged separately; no combined claim)

A variant SUCCEEDS only if BOTH hold:
1. Non-inferiority on bright arterial detection against BRIGHT (seed 1): on the 242 arterial-bright tumours (same set as W6), the paired difference (W6b minus BRIGHT) in detection has a patient-clustered bootstrap 95% CI (B = 2,000) whose lower bound is above -0.10.
2. It passes the forgetting guard as defined in W6: kidney Dice drop at most 0.02 (mean) with the CI lower bound above -0.03, spleen and liver likewise, false positives within 1.25 x baseline median + 0.5 mL, and drift Dice at least 0.95 for every class above the 10 mL floor (the kidney criterion is the organ_kidneys measure).

Possible outcomes are reported as: SUCCEEDS; GUARD ONLY (guard passes, detection non-inferiority fails); DETECTION ONLY (detection holds, guard fails: "fix with forgetting cost" tag kept); NEITHER.
Expectation stated in advance: the full 0.95 drift criterion on all 48 classes is strict (the best W6 arm, DEC, fails 20 classes; the lowest class mean Dice above the floor was 0.75 to 0.81). A partial reduction is plausible but a full pass is uncertain; this is why the graded secondaries below are reported.

## 7. Secondaries (descriptive, no pass/fail beyond the stated)

- Number of classes above the floor failing 0.95, lowest class mean Dice, kidney Dice change, side check, false positives, against BASELINE and against BRIGHT/DEC/PLAIN.
- Detection by contrast group on arterial and portal series, W4b theta and its label, HCC-TACE-Seg detection and Dice (as the W6 tables).
- Over-segmentation tables (no-overlap components, FP volume bands) as for W6.
- A paired comparison of W6b-FULL and W6b-DEC with their W6 counterparts (BRIGHT, BRIGHT-DEC) on the same measures.

## 8. Cost estimate (ESTIMATE from observed W6 timings on an A10 24 GB)

| Item | W6b-FULL | W6b-DEC |
|---|---|---|
| Training 8,000 iterations (W6: 0.62 s/it full, 0.42 s/it decoder-only; teacher forward adds about 0.2 s/it, peak memory about +2 to 3 GB) | about 1.9 h | about 1.4 h |
| WAW-TACE inference + scoring, both halves (W6: about 2.7 h) | 2.7 h | 2.7 h |
| HCC-TACE-Seg pass | 0.2 h | 0.2 h |
| Forgetting-guard inference (W6: about 1.9 h GPU; the drift and false-positive analysis is CPU, about 2 to 3 h wall, can run on any instance) | 1.9 h | 1.9 h |
| Total GPU | about 6.7 h | about 6.2 h |

Both variants: about 13 GPU-h; about 7 h wall on two A10 instances in parallel (plus CPU analysis overlapping). Optional additions, only with approval: a second seed of the variant that looks promising (about +6.5 GPU-h), or a lambda = 5 amendment (about +6.5 GPU-h each). Smoke tests (training data only, below) add about 0.2 GPU-h. Estimates include no queue idle time.

## 9. Before any GPU use (needs the researcher's sign-off)

1. Researcher approves this PREREG (and the lambda proposal, the loss form, K = 16, the two variants, one seed).
2. `scripts/w6b_train.py` and tests written, hashed in `results/w6b_distill/code_hashes.txt`; the shared W6 hash files unchanged.
3. Smoke tests on training data only: (a) 50 iterations per variant run, all losses finite; (b) at iteration 0 the distillation loss is 0 up to numerical noise and its gradient is 0; (c) the teacher's parameters do not change and are on no optimiser; (d) the K-class prompting returns logits for the sampled classes; (e) the measured s/it and memory; (f) the task-loss part is byte-comparable to the W6 loss on the same batch. Results reported before the real runs.
4. No evaluation image (WAW-TACE, guard sets, HCC-TACE-Seg) is opened for design; lambda, K and the loss form are fixed here.

## 10. Rules carried over

Pre-register before analysing; report negative results; deviations logged with timing (DEVIATIONS file); never overwrite a running script in place (write a new file, switch at the next arm); no IPs in committed files; no change to the frozen analysis; instances are never terminated by the assistant. A null or mixed result is reported as such, and a guard failure keeps the "fix with forgetting cost" tag.
