# Pre-specified within-patient test of the "brighter-than-liver tumours are missed" hypothesis
Written before the contrast values of the candidate acquisitions were computed (the tumour Dice per candidate acquisition had already been seen, in `cache/pulled/clean_eval.csv`; the contrast had not).

- **Hypothesis (from the exploratory sign-of-contrast look in `SUMMARY.md` section 3):** within one patient, the acquisition in which the tumour is brighter relative to the liver gives the lower tumour Dice.
- **Data:** the 11 AMBIGUOUS cases of `convert_heldout_clean.py` (the segmentation fits two or three acquisitions of the same patient equally well by position), one clean volume per candidate acquisition.
- **Contrast definition (fixed):** signed contrast = mean HU of GT tumour voxels (seg 2) minus mean HU of non-tumour GT liver voxels (seg 1), both on annotated slices of that acquisition's volume.
- **Unit of analysis:** a case counts only if its candidate acquisitions differ in tumour Dice by at least 0.10. For each such case take the acquisition with the highest signed contrast and the acquisition with the lowest.
- **Result statistic:** the number of such cases in which the higher-contrast acquisition has the LOWER Dice, with a two-sided exact binomial (sign) test against 50%. No other tests, covariates or thresholds.
- **Stated limits:** at most 11 cases; same patients as the exploratory finding (different images); acquisition number is not a phase label; the contrast difference between acquisitions may reflect timing, registration or reconstruction differences, not only phase.
