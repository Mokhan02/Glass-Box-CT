# Glass-Box CT research log

Append-only record of research results on branch `research/roadmap`. Past entries are never edited; a correction is a new dated entry that references the old one. Every number links to the file that produced it. Research use only; nothing here is about patients or clinical use.

## 1. Status

| Workstream | Stage | Last result | Next step | Blocking on |
|---|---|---|---|---|
| W1 training-data contrast audit | Extraction done for CRML-CT; Task03 downloaded, being verified and extracted; comparison not yet run | HCC reproduction passed: 86/86 lesions, max difference 0.0000 HU ([hcc_repro.log](results/w1_training_audit/hcc_repro.log)) | Task03 inventory, then contrast, then `scripts/w1_compare.py` (pre-registered rule) | Task03 SHA-256 and extraction finishing |
| W2 kidney replication | Gate 1 (training-data overlap) reported; Gate 2 (feasibility, ground truth only) in progress | C4KC-KiTS may have been seen in training through AbdomenCT-1K (section 4) | Draft `results/w2_kidney/PREREG.md`, ground-truth lesion and contrast counts, report bright-lesion count | Approval of the PREREG before any model run |
| W3-W7 | Not started | - | - | W1 and W2 |

## 2. Headline results so far (hackathon-era, model NV-Segment-CT / VISTA-3D `vista3d132`, frozen)

All from [results/SUMMARY.md](results/SUMMARY.md) unless another file is named. CIs are patient-level bootstrap 95%. Tags: [PRE-REGISTERED] = pre-registered rule run as written (including runs with a logged deviation); [EXPLORATORY] = exploratory or descriptive, with no pre-registered test.

Cohort and quoting rules: the 89-case clean cohort is quoted at -21% [-30%, -15%]; -19% [-28%, -13%] is the UID route (75 cases) only. Recalibration is quoted from the nested cross-validated analysis, never the in-sample choice.

| # | Finding | Numbers | Tag | Source |
|---|---|---|---|---|
| 1 | One-sided under-measurement of hepatic tumours (sum of diameters) | UID route -19% [-28%, -13%]; clean-89 -21% [-30%, -15%]; detected lesions -11.4 mm, no over-estimates; about 1 in 5 lesions largely missed; Dice 0.551 (single-acquisition-81) | [EXPLORATORY] (descriptive) | [SUMMARY s1](results/SUMMARY.md) |
| 2 | The encoder carries the missed tissue (voxel probe, "lost late") | mid-encoder detection 0.587 [0.551, 0.656] (all-103) vs random-init 0.079 [0.060, 0.090] vs raw HU 0.114 [0.050, 0.146]; median logit of missed voxels -2.59; clean cohorts 0.604 and 0.627 | [PRE-REGISTERED] (with logged deviations #2, #3) | [SUMMARY s2](results/SUMMARY.md), [voxel_probe](results/voxel_probe/) |
| 3 | Brightness causes misses (contrast counterfactual, clean-89) | dark to bright: coverage change -0.208 [-0.284, -0.098], 56 of 62 in the hypothesised direction; bright to dark passes narrowly: +0.238 [+0.001, +0.563], 13 of 14 non-tied | [PRE-REGISTERED] | [SUMMARY s3](results/SUMMARY.md), [contrast_cf](results/contrast_cf/) |
| 4 | A re-fitted readout does not beat a threshold shift | re-fit Dice 0.586 [0.524, 0.647] vs matched threshold shift 0.635 [0.578, 0.689]; re-fit minus threshold -0.049 [-0.073, -0.026]; both over-estimate by >= 20% in 13.5% [6.7, 21.3] of cases (5-fold patient-level cross-validation) | [PRE-REGISTERED] | [SUMMARY s4](results/SUMMARY.md), [readout_refit](results/readout_refit/) |
| 5 | Direct logit attribution: 8 of 48 decoder channels carry the missed-vs-detected gap | 0.77 [0.76, 0.78] of the gap; literal pre-registered verdict "no" on the sign criterion (deviation 1), sign-aware reading post hoc | [PRE-REGISTERED] | [SUMMARY s6](results/SUMMARY.md), [A_SUMMARY](results/mech/A_SUMMARY.md) |
| 6 | Brightness lowers those channels within voxels of the same outcome | detected voxels -0.90 [-1.48, -0.34] logit; missed voxels -1.45 [-1.97, -0.96] | [EXPLORATORY] | [SUMMARY s6](results/SUMMARY.md), [A_strat_exploratory](results/mech/A_strat_exploratory.json) |
| 7 | Patching: the dark tumour's own mid-encoder features restore coverage | 0.79 [0.71, 0.85] (bottleneck 0.66 [0.55, 0.76]); shell-only 0.06, far context 0.03; 32 eligible lesions | [PRE-REGISTERED] | [SUMMARY s6](results/SUMMARY.md), [D_SUMMARY](results/mech/D_SUMMARY.md) |
| 8 | LEACE erasure of the tumour-vs-liver direction at the mid-encoder | lesion Dice -0.33 [-0.38, -0.29] on 78 detected lesions (65% still detected); random direction -0.002; local-HU direction -0.045; liver Dice +0.002; verdict USED | [PRE-REGISTERED] (first run invalid and discarded, deviation 4) | [SUMMARY s6](results/SUMMARY.md), [B_SUMMARY](results/mech/B_SUMMARY.md) |
| 9 | Bridge: brightness lowers that direction in the same tumour | projection bright minus dark -0.93 [-1.22, -0.66], 55 of 62 lower; Spearman with coverage drop 0.74 [0.58, 0.85] | [EXPLORATORY] | [SUMMARY s6](results/SUMMARY.md), [BRIDGE_SUMMARY](results/mech/BRIDGE_SUMMARY.md) |
| 10 | Steering equals a threshold shift | mid -0.006 [-0.013, +0.001], bottleneck -0.003 [-0.009, +0.002] Dice vs matched threshold; alpha at the grid edge in every fold | [PRE-REGISTERED] | [SUMMARY s6](results/SUMMARY.md), [C_SUMMARY](results/mech/C_SUMMARY.md) |
| 11 | Flag-gated recalibration, nested cross-validation (the quoted result) | default median signed error -19.2% [-28.9, -14.3]; always-recalibrated -0.1% [-6.3, +2.7], over-estimated >= 20% in 12.4% [5.6, 20.2], mean Dice 0.640; **gated** -4.5% [-6.2, -2.0], mean absolute error 19.8% [15.2, 25.2], over-estimated >= 20% in 7.9% [3.4, 14.6], mean Dice 0.645. The pre-registered shipping rule is NOT met in this analysis (the over-estimation interval includes zero) | [EXPLORATORY] (sensitivity analysis, not pre-registered) | [SUMMARY s7](results/SUMMARY.md), [trust/](results/trust/) |
| 12 | Quality flag | cross-validated AUROC 0.925 [0.860, 0.980] | [EXPLORATORY] (descriptive) | [SUMMARY s5](results/SUMMARY.md) |
| 13 | Contrast-phase heuristic | UNVALIDATED: agreement with reference-contrast sign 60% (49% to 70%), kappa 0.04 (-0.16 to 0.24) | [PRE-REGISTERED] rule, descriptive evaluation | [SUMMARY s7](results/SUMMARY.md), [PHASE_RESULTS](results/trust/PHASE_RESULTS.md) |

What cannot be said (from [SUMMARY](results/SUMMARY.md)): anything about patients or outcomes; anything about phase; that the threshold shift generalises beyond HCC-TACE-Seg.

## 3. W1: training-data contrast audit

**Question.** Are liver tumours that are brighter than the surrounding liver rare among the public liver-tumour labels VISTA-3D was trained on, compared with the held-out HCC cohort?
**Pre-registration.** [PREREG.md](results/w1_training_audit/PREREG.md), committed 9e1a515. Decision rule: d = p_HCC - p_train (bright = tumour minus liver-shell contrast > 0 HU, two largest lesions >= 10 mm per case, pooled primary), margin 0.10, at least 200 training lesions; applied in order: BRIGHT RARE IN TRAINING (CI lower > 0 and d >= 0.10), REVERSED (CI upper < 0), NO GAP (whole CI within +/-0.10), INCONCLUSIVE.
**Named confounders:** contrast phase and tumour type (metastases vs HCC).

Log (append-only):

| Date | What was done | Result | Commit |
|---|---|---|---|
| 2026-10-12 | Pre-registration approved and committed before any download | - | 9e1a515 |
| 2026-10-12 | Dataset records: MSD Task03 licence CC-BY-SA 4.0, 131 labelled and 70 unlabelled cases, labels 1 liver and 2 cancer; CRML-CT (TCIA Colorectal-Liver-Metastases) CC BY 4.0, 197 patients | matches the PREREG table | 6f4f0ce ([DATASETS.md](results/w1_training_audit/DATASETS.md)) |
| 2026-10-12 | Shared lesion-contrast function with phantom tests; HCC reproduction script | tests pass | 6f4f0ce |
| 2026-10-12 | CRML-CT segmentation survey (metadata only) and patient inventory | 197 patients, 0 excluded; label-swap guard clean; 408 lesions >= 10 mm (194 patients), 301 primary (two largest per case) | 6f4f0ce ([survey](results/w1_training_audit/crml_series_survey.csv), [inventory](results/w1_training_audit/crml_inventory.csv)) |
| 2026-10-13 | Parity review found the first reference left other tumours and vessels in the CRML-CT liver reference; fixed (deviation 3), first lesion table superseded, no share had been computed | contrast moved by > 1 HU in 72 of 408 lesions (52 of 301 primary) | 5565c4f |
| 2026-10-16 | Lesion selection verified unchanged by the fix (per-patient primary counts identical); vessel variants added; class changes between variants counted before any share | CRML-CT: primary vs no-hepatic-vein 2 of 301 flips; primary vs no-vessel-removal 0 of 301 ([counts](results/w1_training_audit/crml_primary_counts_before_after_dev3.csv)) | 5565c4f |
| 2026-10-16 | HCC reproduction on the GPU cloud instance (HCC volumes live on its persistent filesystem): shared function vs exp8 on the UID route | 86 of 86 lesions reproduced, 0 missing, 0 extra, max difference 0.0000 HU and 0.0000 mm, passes the 0.1 HU bar; HCC no-vessel-removal variant: 0 of 86 class changes ([log](results/w1_training_audit/hcc_repro.log), [variants](results/w1_training_audit/hcc_lesions_variants.csv)) | this commit |
| 2026-10-16 | Task03 downloaded on the instance (28,925,891,584 bytes, equal to the source object size); `tar -tf` exit 0 (672 entries) | SHA-256 and extraction pending | - |
| 2026-10-16 | Task03 extractor, comparison script and deviation 4 committed before any Task03 contrast or any share | - | 4e56182 |
| 2026-10-16 | Rows above marked "this commit" (HCC reproduction, W2 Gate 1) were committed together with this log | - | 251aa61 |

**Deviations** ([DEVIATIONS.md](results/w1_training_audit/DEVIATIONS.md)):
1. CRML-CT uses only the MANUAL "Segmentation" series (AI and AI-corrected series are model output); roles read from segment labels, not order.
2. CRML-CT patient exclusion criteria fixed before any contrast.
3. Reference parity with exp8: Liver minus all tumour minus Portal vein is the primary reference (hepatic-vein and no-vessel-removal variants are sensitivities); decided 2026-10-16 before any share.
4. MSD Task03 loader and exclusion criteria fixed before any Task03 contrast.

**Verdict.** None yet.

## 4. W2: kidney replication (C4KC-KiTS)

**Question.** Given a tumour's location, is coverage or RECIST under-measurement worse for kidney tumours brighter than their organ, as for HCC?
**Pre-registration.** Not yet drafted. Gates: (1) training-data overlap, (2) feasibility from ground truth only, (3) model runs after approval.

Log (append-only):

| Date | What was done | Result | Commit |
|---|---|---|---|
| 2026-10-16 | Gate 1: training-data overlap check, no compute | VISTA-3D's Table 1 (arXiv 2406.05285) lists no KiTS19/21/23. The paper uses C4KC-KiTS (210 scans) as a zero-shot evaluation set. It includes AbdomenCT-1K (1,050 volumes; liver, kidney, spleen, pancreas labels). The AbdomenCT-1K paper (Ma et al., arXiv 2010.14808) says: "our dataset, namely AbdomenCT-1K, consists of 1112 3D CT scans from five existing datasets: LiTS (201 cases), KiTS (300 cases), MSD Spleen (61 cases) and Pancreas (420 cases), NIH Pancreas (80 cases), and a new dataset from Nanjing University (50 cases)", and that KiTS "includes 210 training cases with kidney and kidney tumor annotations and 90 testing cases". C4KC-KiTS is the 210 KiTS19 training cases (TCIA page). So C4KC-KiTS images may have been in VISTA-3D's training pool through AbdomenCT-1K. Which cases cannot be determined: each source is split 64/16/20% at random and no case lists are published. | this commit |
| 2026-10-16 | AbdomenCT-1K tumour handling | Not stated. The paper says "We annotate the liver, kidney, spleen, and pancreas for all cases", lists the original KiTS annotation as "Kidney and tumor", and gives the KiTS annotation in its continual-learning benchmark as "Kidney". Whether KiTS tumours are merged into the kidney label, excluded or separate is unresolved; treat "tumours may have been labelled as kidney in training" as a named confound. The paper also says the KiTS (210) scans "were acquired on the arterial phase". | this commit |
| 2026-10-16 | Prompting | Class 129 (kidney mass) cannot be prompted by label (vendored pipeline excludes 129, 130, 131; [AUDIT.md](results/exp1_audit/AUDIT.md)); it needs point prompts. | this commit |

**Deviations.** None yet.
**Verdict.** None yet.

## 5. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-10-12 | Roadmap work lives on branch `research/roadmap`; `main` and its deployed behaviour are never changed | Hackathon build must stay stable |
| 2026-10-12 | Quote -21% for the 89-case clean cohort and -19% only for the UID route; quote nested numbers for flag-gated recalibration | Keeps every quoted figure tied to the cohort that produced it |
| 2026-10-12 | W1 primary = pooled MSD Task03 + CRML-CT; per-dataset secondary; margin 0.10 set by the roughly 100-lesion HCC sample; the 200-lesion minimum applies to the training side; size strata <30, 30-60, >60 mm; tumour type named as a confounder next to phase | A smaller margin could not be resolved with the HCC sample |
| 2026-10-12 | Primary HCC reference = the 75 UID-route cases (86 lesions) saved by exp8 under the same definition; the 89-case cohort is a later sensitivity analysis | exp8 already saved exactly this contrast definition |
| 2026-10-12 | CRML-CT series = MANUAL expert series only, roles by segment label | The AI series are model output; "Liver" and "Liver Remnant" share a code meaning |
| 2026-10-16 | Primary reference = Liver minus all tumour minus Portal vein (exact exp8 parity); hepatic-vein and no-vessel-removal as sensitivity columns | exp8's label 1 excludes every tumour component and the labelled portal vein, not hepatic veins |
| 2026-10-16 | HCC reproduction bar stays at the pre-registered 0.1 HU; between 0.1 and 1 HU means stop, report and log a deviation | Do not pass a mismatch silently |
| 2026-10-16 | Task03 and the W1 analysis run on the GPU cloud instance (HCC volumes and the persistent filesystem are there); downloads proceed only above about 20 MB/s; nothing left running unattended | Local transfer was too slow; the instance bills while up |
| 2026-10-16 | W2 headline arm = one positive point per lesion at its interior point (max distance to the boundary), no negative points, identical protocol run on the HCC UID cohort; class 26 HCC results stay as the separate reported analysis; point prompts give location, so W2 measures appearance-driven under-segmentation, not detection; the automatic kidney mask is descriptive only | Class 129 cannot be label-prompted; a matched protocol makes kidney vs liver like-for-like |
| 2026-10-16 | All of C4KC-KiTS is labelled "possibly seen in training (images, via AbdomenCT-1K)" in the PREREG and every output; no case matching attempted | Cases cannot be identified from public information |
| 2026-10-16 | The encoder probe and LEACE for W2 are specified exactly as in HCC (class-agnostic encoder) | Keeps the mechanism steps comparable |

## 6. Open questions and limitations

- W1 cannot separate "labels rarely bright" from contrast phase and tumour type; W4 and W5 address phase.
- Which VISTA-3D training cases fell in the 64% training split is not public; W1 audits all labelled cases.
- VISTA-3D's MAISI synthetic-tumour stage is not covered by W1.
- C4KC-KiTS may be in-distribution at the image level, and AbdomenCT-1K's tumour labelling is unresolved.
- C4KC-KiTS is described as arterial phase; phase cannot be labelled on HCC-TACE-Seg.
- Whether the kidney cohort has enough bright lesions for a brightness test is unknown until Gate 2.
