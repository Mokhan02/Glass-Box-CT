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

## Model D: PENDING (running)
