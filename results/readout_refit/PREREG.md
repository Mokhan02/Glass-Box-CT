# Pre-registration: class-26 readout re-fit (written and committed before any fitting)

Status: written after the gate passed (`results/voxel_probe/GATE.md`, commit 43e3baa) and before any decoder_F voxel extraction or fitting for this experiment. Research use only; no clinical claim.

## Question
The voxel probe says the encoder represents missed hepatic-tumour tissue and the loss happens later (governing verdict LOST LATE; median interior-FN logit about -2.5). Does re-fitting only the class-26 readout (48 weights + 1 bias on decoder_F) recover that tissue better than simply shifting the decision threshold?

## Data
- Clean cohort, 89 cases (`results/exp1_audit/hcc_clean_conversion.csv` clean set; same cases as the counterfactual baseline). Folds: cases sorted by id, fold = index % 5.
- Same single-window inference as the voxel-probe extraction, prompting classes [1, 26] in one pass.
- ROI: predicted liver (class 1, p > 0.5) dilated by 10 mm. decoder_F (48 ch, 128^3 grid) stored at voxel level inside the ROI only, fp16. Where windows overlap, the window-averaged F is stored (the logit is linear in F, so the average of w.F equals w.(average F)).
- Check before fitting: w_orig . F_avg reproduces the original class-26 logit on one case (max abs difference < 0.05 logit). If not, stop and log in DEVIATIONS.md.

## Model
- New readout: 48 weights w and 1 bias b; logit = w.F + b. Initialised at w = original class-26 readout vector (computed as M(E_c[26]) from the bundle), b = 0.
- Loss on sampled voxels (per training fold): logistic loss + soft Dice. Sample = all GT tumour voxels, a 5-15 mm peritumoural shell (non-tumour voxels), and random liver voxels (random liver count = 3x the tumour count, per case). Class balance handled by the Dice term, no extra reweighting.
- L2 regularisation toward the original vector: lambda * ||w - w_orig||^2. lambda in {1e-3, 1e-2, 1e-1, 1, 10}; chosen by inner 4-fold patient-level CV on the training folds only (criterion: mean voxel soft-Dice on inner-held-out patients). Optimiser: Adam, lr 1e-2, 300 full-batch steps, fixed.
- Applying it: the swapped readout is patched into the network (class-26 weights replaced, bias applied as a threshold at sigmoid(-b) on the class-26 probability map) and the real pipeline (post-processing, lesion extraction, measurement) is run on held-out folds.

## Evaluation (5-fold, patient-level CV; every number from held-out folds)
Per arm: original readout, threshold shift, re-fit. Report with patient-bootstrap 95% CIs (2000 resamples over cases):
- tumour Dice (per case, mean);
- lesion detection (>= 10% of GT lesion volume covered, GT lesions >= 10 mm);
- sum-of-diameters relative error: median, % of cases with |error| >= 20%, % of cases OVERestimated by >= 20%.

## Comparator
Cross-validated threshold shift of the same kind as the voxel-probe recovery: a single logit threshold t <= 0 for class 26, chosen on training folds only. Matching: t is tuned on the training folds so that its >= 20% over-estimation rate equals the re-fit's over-estimation rate on those same training folds (re-fit predictions on training folds come from the fitted readout; the match is evaluated through the ROI-based pipeline, which is an approximation and is reported as such).

## Decision rule (numbers fixed now)
Primary quantity: per-case held-out tumour Dice, re-fit minus matched threshold shift. Patient-bootstrap (2000 resamples, percentile) 95% CI of the mean difference.
- "BEATS THRESHOLD": the CI excludes zero on the positive side AND the held-out over-estimation rate (>= 20%) of the re-fit does not exceed the threshold arm's by more than 5 percentage points.
- Otherwise: "NO BETTER THAN THRESHOLD". Nothing in between is claimed; a CI excluding zero on the negative side is reported as "WORSE THAN THRESHOLD".
No other criterion, subset or metric can change this verdict. Everything else is descriptive.

## Descriptive extras (not part of the decision)
- Cosine similarity: probe direction (fitted on decoder_F) vs original class-26 readout; re-fit vs original.
- For bright (tumour-minus-liver contrast > 0 HU) vs dark tumours: mean projection of missed and of detected tumour voxels onto the original readout and onto the probe direction.
- Subgroups (bright vs dark, size) are exploratory.

## Outputs (`results/readout_refit/`)
Results table (CSV + markdown), one figure (Dice and SLD bias: original vs threshold vs re-fit), two case overlays of a bright missed tumour before and after the re-fit, DEVIATIONS.md for anything that departs from this file (with timing).
