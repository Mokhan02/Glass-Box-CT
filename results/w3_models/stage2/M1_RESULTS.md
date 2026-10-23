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

## Model D (nnU-Net v2, Task03 fold 0, default inference with mirroring): VERDICT = LOST LATE (the REPRESENTED threshold is met narrowly)
Hook = encoder stage 3 (256 channels, stride 8, probed after a 2x average pool: 12.2 x 12.2 x 16 mm tokens, anisotropic through-plane: a stated limitation); the intervention for M2 will act on the same tensor (propagation to skip 3 and to the deeper layers verified on a real case). Tokens: 56,616 interior-FN tokens from 71 patients; the raw-HU control features of D use a 2x subsample for percentiles (Deviation 8).

| layer | trained network | raw-HU control | random-init network |
|---|---|---|---|
| **mid = stage 3 pooled (primary)** | **0.534 [0.475, 0.613]** | 0.162 [0.074, 0.230] | 0.069 [0.059, 0.083] |
| decoder_F (last decoder stage, 32 ch, pooled x16) | 0.737 [0.650, 0.830] | 0.156 [0.074, 0.218] | 0.096 [0.077, 0.125] |
| bottleneck (stage 5, 320 ch, 24.6 mm; reported only, 7,211 tokens / 62 patients) | 0.167 [0.130, 0.191] | 0.333 [0.152, 0.483] | 0.070 [0.062, 0.079] |
| mid_s4 (stage 4 native, 12.3 x 12.3 x 16 mm; hook sensitivity) | 0.316 [0.283, 0.370] | 0.158 [0.069, 0.233] | 0.059 [0.051, 0.070] |

- **Rule (fixed in the PREREG):** REPRESENTED because R_enc = 0.534 >= 0.50 (point estimate) and its CI lower bound (0.475) is above the raw-HU CI upper bound (0.230). **LOST LATE** because criterion (b) holds: the median tumour margin (z_tumour minus the larger of z_liver and z_background) of interior-FN voxels is **-7.11** (IQR -11.20 to -3.62; 7.1% within one unit below 0), far below -2.0. Criterion (a) does NOT hold, and in the opposite direction: **decoder_F detection is HIGHER than mid by +0.202 [+0.160, +0.250]** (0.737 vs 0.534), i.e. the missed tissue becomes more separable in the late features while the head still does not call it tumour; the verdict rests on (b) alone.
- **Fragility, stated plainly:** the primary point estimate (0.534) clears the 0.50 threshold only narrowly and its CI lower bound (0.475) is below 0.50; the rule uses the point estimate, so the verdict is as pre-registered, but a stricter reading would call D's mid-layer evidence borderline. The **hook sensitivity does not reach the threshold**: stage 4 gives 0.316 [0.283, 0.370] (INCONCLUSIVE: above the HU control's upper bound but below 0.50; with HCC_027 included NEVER REPRESENTED because the HU control's CI widens). The bottleneck is below its raw-HU control. Hook sensitivities are not part of the cross-model reading.
- **Controls:** the trained probe beats the raw-HU control by about 0.37 and the random-init network by about 0.47 at mid (CIs separate; the random-init qualification does not apply).
- **Strata (mid):** small tumours 0.688 [0.573, 0.805] vs raw-HU 0.022 vs random-init 0.071 (2,771 tokens, 34 patients); large tumours 0.526 [0.467, 0.608] vs 0.169 vs 0.068 (53,845 tokens, 37 patients): in D the representation is stronger for small tumours than large ones (the reverse of model A).
- **Sensitivity cohorts (same rule, folds and resamples recomputed):** including HCC_027 (75 cases): LOST LATE, mid 0.537 [0.467, 0.622] vs raw-HU 0.196 vs random-init 0.069; excluding the 6 quality-flagged cases (68 cases): LOST LATE, mid 0.531 [0.473, 0.615] vs raw-HU 0.143 vs random-init 0.067. (`m1/an_D*`.)

## M1 verdicts side by side (the criterion for the cross-model reading is "meets M1" = LOST LATE; M2 is not yet run)
| | A (hook `layers2[0]`) | D (hook stage 3, pooled) |
|---|---|---|
| M1 verdict | **LOST LATE** | **LOST LATE** |
| trained / raw-HU / random-init at mid | 0.802 / 0.165 / 0.063 | 0.534 / 0.162 / 0.069 |
| decoder_F minus mid (criterion a) | +0.001 [-0.101, +0.121] (not met) | +0.202 [+0.160, +0.250] (not met, opposite sign) |
| median interior-FN head margin (criterion b) | -6.34 (met) | -7.11 (met) |
| robust to HCC_027 / quality-flagged exclusion | yes / yes | yes / yes |
| hook sensitivity | hidden state 2: LOST LATE | stage 4: INCONCLUSIVE (0.316) |
Both models meet M1 as pre-registered. Neither shows the decoder LOSING the signal relative to the mid layer; "lost late" is carried by the head margin in both. M2 (necessity) decides whether the mechanism replicates.

## Model A U-Net checkpoint (sensitivity arm, not part of the cross-model reading): LOST LATE, borderline on REPRESENTED
The CLIP-Driven U-Net checkpoint through its shipped pipeline, same cohort (74 cases), hook `backbone.down_tr256` output 0 (256 channels, stride 8, 12 mm); bottleneck = `down_tr512` output 0 (512 channels, also stride 8 in this U-Net), decoder_F = `backbone.up_tr64` (64 channels, pooled x8). Interior-FN tokens: 48,137 from 66 patients.

| layer | trained network | raw-HU control | random-init network |
|---|---|---|---|
| **mid** | **0.522 [0.472, 0.624]** | 0.170 [0.065, 0.231] | 0.110 [0.071, 0.184] |
| bottleneck | 0.631 [0.544, 0.683] | 0.172 [0.063, 0.234] | 0.080 [0.060, 0.130] |
| decoder_F | 0.761 [0.704, 0.827] | 0.177 [0.062, 0.241] | 0.088 [0.063, 0.142] |
Rule: REPRESENTED (0.522 >= 0.50, CI lower 0.472 above the HU control's 0.231; again a narrow pass) and LOST LATE via criterion (b): median margin of interior-FN voxels -3.23 (IQR -5.61 to -1.51; 16.6% within one unit below 0, a smaller margin than the Swin model's -6.3). Criterion (a) not met (decoder_F minus mid = +0.239 [+0.159, +0.280]). Random-init CI separate from the trained CI. Strata (mid): small 0.489 [0.388, 0.651], large 0.524 [0.471, 0.638]. This arm is a sensitivity and is not part of "mechanism replicates".


## Caveats attached to every number above
- The VISTA-3D probe-table reproduction (sanity check 1) is near-exact, not exact, in the current environment: the headline row matches (0.5873) and the verdict is the same, but 47 of 228 compared cells differ (maximum 0.006, in small strata and the 768-d bottleneck), the same disclosed limitation as W2-M DEVIATION 1 (the stored table comes from an earlier solver environment; two analysis environments here give identical tables to each other). Logged as Deviation 8.
- Tokens exist only inside the GT liver + tumour dilated by 10 mm on annotated slices; "missed" is the model's own default prediction; the probe separates TUMOUR from LIVER tokens, so it shows linear separability, not use (M2 tests use).
- The decoder_F "late" layer for model A is the 48-channel decoder output before the class-conditioned head; the lost-late call rests on the head margin criterion (b).
