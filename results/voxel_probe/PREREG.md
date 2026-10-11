# Pre-registration: where does the under-segmentation of held-out hepatic tumours happen?

Written and committed BEFORE any extraction or probe was run for this experiment. Numbers below are fixed. Any deviation is logged
in `DEVIATIONS.md` with its reason; results are reported regardless of direction. No new probe types or feature sets after results.

## Question
NV-Segment-CT under-segments hepatic tumours on held-out HCC-TACE-Seg CT. Does the encoder represent the missed tumour tissue
(so the loss happens later, in the decoder or class head), or does the encoder never represent it as tumour?

## Data (all rows `in_distribution = False`)
- All 103 converted HCC-TACE-Seg cases; each case is one patient (verified: 103 patients, one labelled study each).
- GT: manual SEG (`Liver` = seg 1, `Mass` = seg 2 = tumour). Predicted tumour = class 26 of VISTA3D (`vista3d132`), p > 0.5.
- Label routes reported separately and pooled: `uid` (n=75) and `geometry` (n=28).
- Annotated voxels only: GT exists only on annotated slices (some SEGs label every other slice). Voxels on unannotated slices are
  "unknown"; coverage fractions below are computed over annotated voxels, and tokens with < 50% annotated voxels are dropped.

## Tokens and layers
Same hooks as Phase 2, run on the trained model and on a randomly initialised copy (`torch.manual_seed(0)`):
- `mid` (encoder stage 3, pooled x2): 192-d, 12 mm grid  -- PRIMARY encoder layer for the decision rule
- `bottleneck` (encoder stage 5): 768-d, 24 mm grid     -- reported; resolution loss expected
- `decoder_F` (auto-decoder output, pooled x8): 48-d, 12 mm grid
Tokens are kept only if the token centre lies inside the GT liver mask (liver + tumour) dilated by 10 mm.
Overlapping sliding windows (overlap 0.3) duplicate some locations; all copies are kept and a patient never straddles CV folds.

Per-token masks by fractional coverage (cell = stride^3 voxels in the 1.5 mm preprocessed grid):
- TUMOUR token: GT tumour coverage >= 0.5.   LIV token: GT tumour coverage == 0 and GT liver (non-tumour) coverage >= 0.5.
- All other tokens are dropped (mixed).
- TP: tumour token with predicted-tumour coverage >= 0.5.   FN: tumour token with predicted-tumour coverage < 0.5.
- "interior": distance from the token centre to the nearest non-tumour voxel of the GT > 2 mm (Euclidean, 1.5 mm voxels);
  otherwise "boundary". Interior FN is the main test.
- Resolution loss: for every GT tumour connected component (26-connectivity), it "vanishes" at a layer if no TUMOUR token has its centre inside it.

## Probes (identical across layers and controls)
- 5-fold GroupKFold by patient (fixed order = sorted case id). `StandardScaler` + L2 `LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)`. C is not tuned.
- Training sets capped at 200 tokens per (patient, group), seed 0. Evaluation uses ALL held-out tokens.
- Probe A: TUMOUR vs LIV.   Probe B: TP vs LIV, evaluated on FN tokens.
- Decision threshold: the 95th percentile of LIV scores among the training-fold LIV tokens (95% specificity on training folds).
  Realised specificity on held-out LIV tokens is reported.
- PRIMARY metric: detection rate = share of held-out interior-FN tokens scoring above the threshold (pooled over tokens).
  Also reported: patient-averaged rate, and AUROC (FN vs LIV, all held-out FN tokens).
- CIs: patient-level bootstrap, 1000 resamples, 95% percentile intervals. Number of patients contributing interior-FN tokens is reported per row.

## Controls (same probes, same tokens)
1. Local raw-HU control: 14 features per token = mean, std, p5, p25, p50, p75, p95 of HU inside the token cell, and the same 7 over the 3x-cell cube
   centred on the token (single fixed feature set; HU recovered from the model input by inverting the bundle intensity scaling).
2. Randomly initialised model: same hooks, same probes.
3. Strata: small vs large tumours (median split by GT tumour volume in mm^3), evaluated with probes trained on all patients.

## Head check
- Class-26 logit (decoder_F x class embedding, as averaged by the sliding window) of interior-FN voxels (GT tumour voxels on annotated slices,
  > 2 mm from the GT edge, predicted p <= 0.5): report median, IQR, and the share within one logit unit below 0.
- Threshold recovery: lower the decision threshold on the logit grid {0, -0.5, ..., -4.0}; choose it by maximising mean tumour Dice on TRAINING
  folds only (same 5 folds), apply to held-out folds, report change in held-out mean tumour Dice and in the RECIST sum-of-diameters bias
  (`scripts/recist_shift.py` definitions, detection = >= 10% of lesion volume covered), pooled and by label route.

## Decision rule (fixed)
Let R_enc = detection rate of Probe A on interior-FN tokens at the `mid` layer (trained model), CI_enc its 95% CI;
R_hu = the raw-HU control's rate at the same layer with CI_hu.
- **Encoder represents the missed tissue**: R_enc >= 0.50 AND lower bound of CI_enc > upper bound of CI_hu.
- **Lost late**: the above holds AND at least one of
  (a) decoder_F Probe A detection is lower than R_enc by >= 0.15 with the patient-bootstrap 95% CI of the difference excluding 0, or
  (b) the median class-26 logit of interior-FN voxels is <= -2.0 (far below threshold; > -2.0 = near-miss).
- **Never represented**: lower bound of CI_enc <= upper bound of CI_hu (the encoder does not beat raw HU).
- **Anything else** (for example above the HU control but R_enc < 0.50) is reported as **inconclusive**.
Interpretation note (does not change the rule): if the randomly initialised model's rate has a CI overlapping the trained model's, "represents"
is qualified as "separable from liver in features a random network also provides". The bottleneck is reported but not used for the decision.

## Known limits, stated in advance
~100 patients; treated tumours (many are small or heterogeneous); label-route gap (uid vs geometry tumour Dice 0.585 vs 0.347, partly tumour size);
bottleneck resolution (24 mm cells) is coarse relative to many tumours; the GT is a semi-automatic mask, not pathology; 95%-specificity thresholds are set
on training-fold tokens, so held-out specificity can differ (reported).
