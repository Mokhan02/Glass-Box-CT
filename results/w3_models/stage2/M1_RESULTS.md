# W3 Stage 2, M1: is the missed tumour represented at the hook, and lost late? (PREREG_stage2.md)

Cohort: W3 primary, 74 cases / 85 lesions after Amendment 3 (HCC_027 only in a sensitivity row). Probe = VISTA-3D voxel probe unchanged (5-fold GroupKFold by patient, `StandardScaler` + L2 logistic regression C = 0.1, balanced, training sets capped at 200 tokens per (patient, group), threshold = 95th percentile of the liver-token scores of the training folds). Numbers are the detection rate of interior false-negative tumour tokens (tokens the model's own prediction missed, more than 2 mm inside the GT tumour) at 95% specificity, with patient-level bootstrap 95% CIs (per-experiment seed = SHA-256 of the arm name). Analysis: `scripts/w3s2_m1_analyze.py` (code hashes in `code_hashes_stage2.txt`; extraction and probe-table reproduction as logged in `results/w3_models/DEVIATIONS.md` entry 8).

## Model A (CLIP-Driven Universal Model, Swin UNETR, shipped pipeline): VERDICT = LOST LATE
Hook = `swinViT.layers2[0]` output (192 channels, stride 8, 12 mm tokens; the same tensor M2 will edit; propagation to the skip copy and the deeper layers verified on a real case before extraction). Tokens: 106,892 interior-FN tokens from 73 patients.

| layer | trained network | raw-HU control | random-init network |
|---|---|---|---|
| **mid (primary)** | **0.802 [0.724, 0.855]** | 0.165 [0.097, 0.218] | 0.063 [0.043, 0.078] |
| decoder_F (48 ch, pooled x8) | 0.802 [0.741, 0.877] | 0.161 [0.097, 0.207] | 0.117 [0.087, 0.157] |
| bottleneck (768 ch, 48 mm; reported only, 1,572 tokens / 42 patients) | 0.922 [0.852, 0.963] | 0.394 [0.217, 0.528] | 0.478 [0.328, 0.579] |
| mid_hs2 (hidden state 2, the LayerNorm copy; sensitivity) | 0.798 [0.721, 0.850] | 0.158 [0.094, 0.206] | 0.067 [0.046, 0.082] |

- **Rule (fixed in the PREREG):** REPRESENTED because R_enc = 0.802 >= 0.50 and its CI lower bound (0.724) is above the raw-HU control's CI upper bound (0.218). **LOST LATE** because criterion (b) holds: the median class-27 logit of interior-FN voxels is **-6.34** (IQR -9.65 to -3.38; share within one unit below 0: 6.1%), far below the -2.0 near-miss limit. Criterion (a) does NOT hold: decoder_F detection is not lower than mid (decoder_F minus mid = +0.001 [-0.101, +0.121]); the verdict rests on (b) alone.
- **Controls:** the trained probe beats the raw-HU control by about 0.64 and the random-init network by about 0.74 at mid (CIs far apart; "random-init CI separate from the trained CI", so the qualification "separable in features a random network also provides" does not apply). At the bottleneck the random-init network is also high (0.478) on very few tokens, which is why the bottleneck is not used.
- **Strata (mid, trained vs raw-HU vs random-init):** small tumours 0.481 [0.366, 0.600] vs 0.043 vs 0.067 (5,741 tokens, 36 patients); large tumours 0.820 [0.743, 0.872] vs 0.172 vs 0.062 (101,151 tokens, 37 patients). The representation is weaker for small tumours (their tokens are mostly mixed or dropped at 12 mm), but still far above both controls; at decoder_F small tumours reach 0.747.
- **Sensitivity cohorts (same rule, folds and resamples recomputed):** including HCC_027 (75 cases): LOST LATE, mid 0.788 [0.704, 0.840] vs raw-HU 0.180 vs random-init 0.059; excluding the two quality-flagged cases HCC_022 and HCC_043 (72 cases): LOST LATE, mid 0.795 [0.711, 0.848] vs 0.157 vs 0.069. Hidden-state-2 probe: LOST LATE (0.798). (`m1/an_A*`.)

## Model D (nnU-Net v2, Task03 fold 0, stage 3 hook): PENDING (extraction running)

## Model A U-Net checkpoint (sensitivity arm, not part of the cross-model reading): PENDING

## Caveats attached to every number above
- The VISTA-3D probe-table reproduction (sanity check 1) is near-exact, not exact, in the current environment: the headline row matches (0.5873) and the verdict is the same, but 47 of 228 compared cells differ (maximum 0.006, in small strata and the 768-d bottleneck), the same disclosed limitation as W2-M DEVIATION 1 (the stored table comes from an earlier solver environment; two analysis environments here give identical tables to each other). Logged as Deviation 8.
- Tokens exist only inside the GT liver + tumour dilated by 10 mm on annotated slices; "missed" is the model's own default prediction; the probe separates TUMOUR from LIVER tokens, so it shows linear separability, not use (M2 tests use).
- The decoder_F "late" layer for model A is the 48-channel decoder output before the class-conditioned head; the lost-late call rests on the head margin criterion (b).
