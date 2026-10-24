# W3 Stage 2, M2: is the tumour-vs-liver direction necessary? Cross-fitted rank-1 LEACE at the hook (PREREG_stage2.md, mech B margins)

Cohort: W3 primary, 74 cases (HCC_027 run too, sensitivity only). Population = lesions the unhooked model detects (coverage >= 0.10) at baseline; outcome = change in lesion Dice (Dice within 10 mm of the lesion, the mech B definition) relative to the in-session unhooked base pass. Each case runs only under the eraser of its own cross-fitting fold (fold = index in the sorted primary list mod 5, fitted on tumour and liver tokens of the other folds). Controls at the same rank: a random unit direction (orthogonal projection through the training-token mean) and LEACE for the local-HU concept. Hook: model A `swinViT.layers2[0]` output (both consumers verified to change on a real case); model D `encoder.stages[3]` output (mirroring on). Case-level bootstrap 95% CIs, per-arm seeds. Determinism checks (all 75 cases): the base pass reproduces Stage 1 lesion coverage to <= 2.8e-4 (limit 1e-3) with no detection-flag mismatch; the identity hook equals base exactly (3 cases).

## Model A (shipped pipeline): USED, but "not testing necessity" is flagged by the pre-registered probe check
55 detected lesions in 51 cases.
| change in lesion Dice (hooked minus base) | mean [95% CI] |
|---|---|
| **LEACE (tumour-vs-liver direction)** | **-0.124 [-0.155, -0.094]** |
| random direction | -0.002 [-0.004, -0.001] |
| local-HU LEACE | +0.035 [+0.020, +0.052] |
| LEACE minus random | -0.122 [-0.152, -0.094] |
| LEACE minus HU | -0.160 [-0.197, -0.122] |
Fraction of baseline-detected lesions still detected: LEACE 83.6%, random 100%, HU 98.2%. Mean coverage change: LEACE -0.110, random -0.002, HU +0.033. **Liver guard:** mean change in liver Dice under LEACE -0.0001 [-0.0015, +0.0015] (the organ is untouched; "too blunt" threshold -0.05).
- **Rule (mech B margins):** USED = mean LEACE change below 0 with a CI excluding 0 AND both paired differences with CIs entirely below 0: all met. No "too blunt", no "erasure improves".
- **Probe check (mech B):** a probe fitted on erased training-fold tokens and tested on erased held-out tokens: AUROC 0.586 [0.520, 0.666] after erasure (0.956 [0.942, 0.966] before). The pre-registered criterion is "near chance iff the CI upper bound <= 0.60"; the upper bound 0.666 exceeds it, so the run is flagged **"not testing necessity"** and, by the cross-model reading fixed in the PREREG ("meets M2" = USED, not too blunt, probe check passed), **model A does NOT meet M2 under the literal rule** although the erasure removes most of the linear signal (point estimate 0.586). Reported as it falls; whether a point estimate of 0.586 with this CI is "near enough" is the researcher's call, not mine to change after the fact.

**Interpretation note (labelled as interpretation; NOT a rule and no rule was changed):** the flag means the cross-fitted erasure left some linear tumour-vs-liver signal in the held-out tokens (AUROC 0.586 rather than 0.50). Residual signal means the erasure was incomplete, and an incomplete erasure can only leave more of the concept available to the network, so it can only shrink the measured effect on lesion Dice. Read this way, the -0.124 is a conservative estimate of what a complete erasure of the direction would cost, not an inflated one. This reading is an inference about the direction of the bias, not something the experiment tested; the pre-registered classification stands: model A is reported as USED, flagged "not testing necessity", and does not count toward "meets M2" in the cross-model rule.

**EXPLORATORY note (not a pre-registered test; no verdict depends on it):** in model A, erasing the LOCAL-HU direction (the cell-mean-HU concept) improved lesion Dice by +0.035 [+0.020, +0.052] and coverage by +0.033, with 98% of baseline-detected lesions still detected. This is consistent with the brightness hypothesis (removing brightness information from the tokens helps the model's handling of tumours that are brighter than the liver), but it was not tested as such: M2's controls were designed to show specificity of the tumour-vs-liver erasure, and the HU erasure is a control arm. In model D the same control did nothing (+0.006 [-0.028, +0.033]).

**Scope of comparison:** these M2 numbers are reported separately and are NOT compared in magnitude with the VISTA-3D mechanism-B erasure (-0.33): that experiment used a different inference route (the class-26 automatic pass with prompts [1, 26] and a liver-ROI readout) and a different lesion-Dice definition context, so the sizes are not comparable.

## Model D (nnU-Net, mirroring on): USED, no flags: MEETS M2
55 detected lesions in 51 cases.
| change in lesion Dice (hooked minus base) | mean [95% CI] |
|---|---|
| **LEACE (tumour-vs-liver direction)** | **-0.121 [-0.150, -0.095]** |
| random direction | -0.001 [-0.003, +0.000] |
| local-HU LEACE | +0.006 [-0.028, +0.033] |
| LEACE minus random | -0.120 [-0.150, -0.093] |
| LEACE minus HU | -0.127 [-0.163, -0.094] |
Fraction of baseline-detected lesions still detected: LEACE 85.5%, random 100%, HU 94.5%. Mean coverage change: LEACE -0.125, random -0.001, HU +0.008. **Liver guard:** mean change in liver Dice under LEACE -0.0028 [-0.0057, -0.0003] (far inside the -0.05 limit). **Probe check:** post-erasure AUROC 0.535 [0.500, 0.584] (before erasure 0.897 [0.869, 0.921]): the CI upper bound 0.584 <= 0.60, so the probe check passes ("near chance"). Verdict USED, not too blunt, probe check passed: **meets M2**. The base pass reproduces Stage 1 lesion coverage to <= 8e-5 over the 86 lesions with no detection-flag mismatch; the identity hook equals base exactly.

## Sensitivity cohorts (same rule, same erasers; the probe check always uses the primary 74-case token set)
| arm | verdict | LEACE change in lesion Dice | LEACE minus random | LEACE minus HU |
|---|---|---|---|---|
| A, with HCC_027 (56 lesions) | USED, flagged | -0.122 [-0.153, -0.093] | -0.120 [-0.151, -0.093] | -0.157 [-0.195, -0.120] |
| A, excluding HCC_022 and HCC_043 (54 lesions) | USED, flagged | -0.125 [-0.156, -0.097] | -0.123 [-0.155, -0.095] | -0.164 [-0.203, -0.126] |
| D, with HCC_027 (56 lesions) | USED, meets M2 | -0.119 [-0.146, -0.092] | -0.118 [-0.146, -0.093] | -0.126 [-0.160, -0.094] |
| D, excluding the 6 flagged cases (52 lesions) | USED, meets M2 | -0.119 [-0.150, -0.094] | -0.118 [-0.149, -0.092] | -0.123 [-0.157, -0.092] |

## M2 verdicts side by side, and the fixed cross-model reading (no rule changed)
| | A | D |
|---|---|---|
| M1 | LOST LATE | LOST LATE |
| M2 verdict | USED | USED |
| lesion Dice change under LEACE | -0.124 [-0.155, -0.094] | -0.121 [-0.150, -0.095] |
| random / HU control | -0.002 / +0.035 | -0.001 / +0.006 |
| liver Dice change | -0.0001 | -0.0028 |
| post-erasure probe AUROC [95% CI] | 0.586 [0.520, 0.666] | 0.535 [0.500, 0.584] |
| flags | "not testing necessity" | none |
| meets M2 (USED, not too blunt, probe check passed) | **no (flag)** | **yes** |
- **Reading as pre-registered (confirmed by the researcher):** MECHANISM REPLICATES requires both models to meet M1 AND M2: not met, because A is flagged. The reading is **PARTIAL REPLICATION**: model D meets M1 and M2, model A meets M1 and shows the USED effect but is flagged on the probe check.
- **Descriptive:** the erasure costs both models almost exactly the same lesion Dice (-0.124 vs -0.121), is specific to the tumour direction (random and HU erasers do nothing), spares the liver, and leaves 84% (A) and 86% (D) of baseline-detected lesions detected. The A interpretation note above applies to the flag only; it does not change the classification. M3 (brightness counterfactual) and M4 (threshold-shift generality) are not yet run.
