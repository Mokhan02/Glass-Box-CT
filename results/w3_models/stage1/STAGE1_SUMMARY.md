# W3 Stage 1 summary (HCC UID cohort, 75 cases / 86 lesions: 29 bright, 57 dark after the HCC_019 correction)

Generated analyses: `STAGE1_SUMMARY/w3_results.md` (A, D and the VISTA-3D reference row together, pinned per-arm seeds); per-model folders hold their own sensitivities. Gaps = dark minus bright; verdict margin 0.10; CIs are case-level bootstrap 95%.

| model | detection gap | Dice gap | label | bright / dark detection | bright / dark mean Dice | median case sum-of-diameters error |
|---|---|---|---|---|---|---|
| A: CLIP-Driven Universal Model, Swin UNETR, shipped pipeline | +56.6 pts [+37.2, +74.9] | +30.6 pts [+21.4, +39.2] | **GAP (detection and Dice)** | 27.6% / 84.2% | 0.133 / 0.439 | -29.4% |
| D: nnU-Net v2, Task03 fold 0, 1000 epochs, default inference | +61.8 pts [+41.3, +78.3] | +41.0 pts [+29.9, +51.4] | **GAP (detection and Dice)** | 24.1% / 86.0% | 0.117 / 0.528 | -34.1% |
| VISTA-3D class 26 (reference row, reproduces the stored numbers) | +44.8 pts [+22.9, +64.5] | +41.9 pts [+28.0, +54.2] | GAP (detection and Dice) | 51.7% / 96.5% | 0.245 / 0.664 | -19.4% |

**Replication rule (both A and D "GAP (detection and Dice)"): met.** Size-adjusted: A Dice +28.3 / detection logit +2.55, D Dice +37.1 [+25.6, +48.9] / detection logit +2.98 [+1.92, +4.71]; none of the adjusted effects disappears. Model A's shipped and raw variants agree (post-processing inert); nnU-Net's default has no post-processing beyond its own inference (mirroring, step 0.5, no postprocessing file).

## Alignment gate (Amendment 2 with the Dice-maximising translation test, Deviation 5), every case
- Model A (swin): 75 of 75 evaluated; **1 MISALIGNED, HCC_027_967743**; 0 model-side; centroid offset > 15 mm in none (max 13.4 mm, HCC_022). Status OK (limit 2).
- Model D: **1 MISALIGNED, the same HCC_027_967743**; the 6 cases that fail the committed centroid rule (HCC_022, 028, 070, 074, 077, 090) are **MODEL-SIDE in the researcher's sense**: the best translation is at most 1.6 mm per axis and raises organ Dice by at most 0.016 (stray non-liver blobs in 028, 070, 074, 090; real liver failures in 022 and 077). Status OK (limit 2). Quality-flagged (kept in the primary, sensitivity excludes them): those 6 for D; HCC_022 and HCC_043 for A.
- **HCC_027_967743 is MISALIGNED in both models, independently, by the same through-plane shift** (best translation -10.0 mm = 4 slices when the search window is widened; organ Dice 0.854 -> 0.947 for A and 0.854 -> 0.948 for D; both models place the liver on slices 18-78 while the GT liver spans 13-74). The source SEG for this case has zero offset against its CT (all 316 frames match their referenced instances, 0.0 mm), so this is not the HCC_019 converter bug; it looks like a label displaced by 4 slices at the source. It was NOT flagged by the earlier 89-case data-quality screen (its z search was limited to +-3 slices). It stays in the primary for now; see the decision below.

## Sensitivities (verdicts unchanged everywhere)
- A: as converted +54.2 / +29.4 GAP; excluding HCC_022 and HCC_043 +57.3 / +30.0 GAP (26 bright / 57 dark); excluding those two and HCC_027 +60.2 / +30.6 GAP (25 / 57).
- D: as converted +59.5 / +39.7 GAP; excluding the 6 quality-flagged +61.5 / +41.3 GAP (26 / 52); excluding the 6 and HCC_027 +64.5 / +42.2 GAP (25 / 52).
- Model A repeat (organ masks saved): the lesion table is not bit-identical (max |coverage| difference 0.0003, |Dice| 0.0005, no detection flips, no group changes) because of GPU non-determinism; the verdict and gaps are identical to four decimals (`A_swin_hcc/repeat_v3/`). The first run's table is the reported one.
