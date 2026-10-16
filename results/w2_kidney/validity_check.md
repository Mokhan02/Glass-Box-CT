# W2 validity check: overlays and component audit (16 lesions rerun with masks saved)

Selection fixed before the rerun ([validity_selection.csv](validity_selection.csv)): per organ, the 2 bright + 2 dark lesions with the largest relative diameter error, plus the 2 bright + 2 dark closest to their group's median error. Determinism check: all 16 lesions reproduced the Gate 3 CSV exactly (coverage, Dice, diameter, point) ([validity_rerun.log](validity_rerun.log)). Per-lesion audit: [validity_audit.csv](validity_audit.csv); overlays: [overlays/](overlays/) (green = GT lesion, red = prediction, cyan = GT organ label, yellow + = prompt point).

## Result: no measurement bug; the predictions really are that large
- In all 16 lesions the measured component is the component that contains the prompt point, and the recomputed diameter equals the Gate 3 value. CT, GT and prediction share affine, spacing and shape in all 16.
- Extra (non-overlapping) components exist in some lesions but never produce the measured diameter; the point-component diameter equals the measured diameter in every case.
- The prediction is essentially the organ the point sits in: the median fraction of predicted voxels lying inside the lesion is small, most of the rest lies inside the organ label (liver or kidney), and the predicted volume is many times the lesion volume (table below).

| organ | lesions | GT diameter (mm, median) | measured = point-component diameter (mm) | pred / lesion volume | share of pred inside lesion | inside organ label (excl. lesion) | outside organ label | pred components |
|---|---|---|---|---|---|---|---|---|
| hcc_130 largest-error | 4 | 23 | 196 | 208.5x | 0.0% | 96.0% | 3.0% | 1 |
| hcc_130 typical | 4 | 66 | 189 | 14.4x | 7.0% | 88.0% | 5.0% | 2 |
| kidney_129 largest-error | 4 | 14 | 86 | 239.3x | 0.0% | 98.0% | 1.0% | 2 |
| kidney_129 typical | 4 | 38 | 70 | 5.3x | 24.0% | 76.0% | 0.0% | 1 |

(Medians over the 8 lesions per row; the typical rows are the pre-specified near-median lesions, the largest-error rows the pre-specified outliers.)

## What this means for the W2 verdicts (validity caveat)
- Coverage is at ceiling because the point-prompted class 129/130 output is close to a whole-organ mask, which contains the lesion by construction. The headline coverage gap therefore does NOT measure appearance-driven under-segmentation of tumours; it mostly measures how much of the organ is returned. The Gate 3 verdicts (kidney NO MEANINGFUL GAP with reversed direction; HCC class 130 INCONCLUSIVE) stand as the pre-registered result but carry this validity caveat.
- Dice and RECIST over-estimation in Gate 3 are real properties of these predictions (not a bug): a point inside the organ returned most of the organ.
- Open question, not decided here: the PREREG protocol passes the class (129 or 130) together with the point. The official zero-shot interactive use of point prompts may be point-only (no class); whether the class-plus-point combination returns an organ-level mask by design or because of how it was called has not been tested. Any change to the protocol needs sign-off and a new pre-registration; none is applied.
- Any Dice-by-brightness analysis of these predictions would be EXPLORATORY and would carry the same caveat.
