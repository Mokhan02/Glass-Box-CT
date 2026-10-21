# W3 Stage 1 summary (HCC UID cohort after the data-quality rules: 74 cases / 85 lesions, 28 bright / 57 dark)

HCC_019_773068 is scored against its corrected label (Amendment 1); HCC_027_967743 is excluded as a source-label error (Amendment 3, `results/dq_hcc_labels/hcc027/DQ_HCC027.md`) and appears only in the sensitivity row "including data-quality excluded". Analyses: `STAGE1_SUMMARY/w3_results.md` (A, D and the VISTA-3D reference row together, pinned per-arm seeds); per-model folders hold their own sensitivities. Gaps = dark minus bright (positive = bright worse); verdict margin 0.10; case-level bootstrap 95% CIs.

| model | detection gap | Dice gap | label | bright / dark detection | bright / dark mean Dice | median case sum-of-diameters error |
|---|---|---|---|---|---|---|
| A: CLIP-Driven Universal Model, Swin UNETR, shipped pipeline | +59.2 pts [+39.3, +77.6] | +31.1 pts [+21.3, +40.7] | **GAP (detection and Dice)** | 25.0% / 84.2% | 0.128 / 0.439 | -29.9% |
| D: nnU-Net v2, Task03 fold 0, 1000 epochs, default inference | +64.5 pts [+44.1, +80.7] | +41.9 pts [+30.9, +52.5] | **GAP (detection and Dice)** | 21.4% / 86.0% | 0.109 / 0.528 | -34.3% |
| VISTA-3D class 26 (reference row; reproduces the stored numbers on the published 86 lesions) | +46.5 pts [+24.5, +66.7] | +42.2 pts [+28.2, +53.6] | GAP (detection and Dice) | 50.0% / 96.5% | 0.242 / 0.664 | -19.0% |

**Replication rule (both A and D "GAP (detection and Dice)"): met.** Size-adjusted effects remain for every model (A Dice +28.9 [+19.1, +37.8], detection logit +2.72 [+1.66, +4.29]; D Dice +38.0 [+27.1, +49.1], detection logit +3.23 [+2.10, +5.09]). Model A's shipped and raw variants agree (post-processing inert). nnU-Net's default inference: test-time mirroring over all flips, sliding window step 0.5, no post-processing file.

## Alignment gate (Amendment 2 with the Dice-maximising translation test, Deviation 5)
- A: only HCC_027_967743 MISALIGNED; D: only HCC_027_967743 MISALIGNED; D's 6 centroid failures (HCC_022, 028, 070, 074, 077, 090) are MODEL-SIDE (best shift <= 1.6 mm, gain <= 0.016): quality-flagged, kept in the primary, excluded in the sensitivity (A: HCC_022 and HCC_043). 89-case re-screen with both models over +-40 voxels per axis: only HCC_027, flagged by both.

## Sensitivities (all GAP (detection and Dice))
- A: including HCC_027 +56.6 / +30.6 (29/57); as converted (HCC_019 displaced) +56.8 / +29.9 (27/58); excluding HCC_022 and HCC_043 +60.2 / +30.6 (25/57).
- D: including HCC_027 +61.8 / +41.0; as converted +62.3 / +40.6; excluding the 6 flagged +64.5 / +42.2 (25/52).
- VISTA reference row: including HCC_027 +44.8 / +41.9; as converted +43.0 / +40.2.
- Model A repeat (organ masks saved): lesion table not bit-identical (max |coverage| difference 0.0003, |Dice| 0.0005, no flips); identical gaps to four decimals; the first run's table is the reported one.
- The model A kidney arm was skipped by the researcher; the model A U-Net sensitivity is deferred to Stage 2.
