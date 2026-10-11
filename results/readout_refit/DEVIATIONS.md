# Deviations / clarifications for the readout re-fit

Each entry states when it was written relative to any fitting. No fitting has been run at the time of entries 1-3.

## 1. Sampling counts the PREREG left open (written before any fit; extraction was running)
PREREG fixed "all GT tumour voxels, a 5-15 mm peritumoural shell, random liver voxels (3x the tumour count)". Not specified there, fixed now:
- peritumoural shell: known (annotated-slice), non-tumour ROI voxels 5-15 mm from the GT tumour; count = min(available, max(3000, 1x tumour voxel count)) per case;
- random liver voxels: known GT-liver (label 1) ROI voxels, non-tumour; count = min(available, max(3000, 3x tumour voxel count)) per case;
- sampling seed 0. No tumour voxel is subsampled.

## 2. Evaluation space and pipeline (written before any fit)
PREREG says the swapped readout runs through "the real pipeline". Implementation: the class-26 logit is recomputed from the stored window-averaged decoder_F only inside the ROI (predicted liver dilated 10 mm); outside the ROI every arm keeps the original logit (threshold 0). All three arms (original, threshold shift, re-fit) are evaluated identically in the preprocessed 1.5 mm RAS grid on annotated slices, cropped to the bounding box of (ROI union GT tumour union original positives) so no original positive is dropped. This is the same space the voxel-probe analysis used for its own Dice, but it is not the original-geometry space of `eval_heldout.py`, so absolute numbers differ slightly from those. Verified on every case that w_orig . F reproduces the network logit (max abs error logged in the extraction log).
The threshold arm shifts the threshold inside the ROI only (same as the re-fit); the voxel-probe threshold-recovery applied it globally.

## 3. Threshold grid and matching (written before any fit)
Matching grid for the comparator: t in {0, -0.25, ..., -5.0} (logit units). On training folds, the re-fit's over-estimation rate (>= 20%) is measured in-sample on the training cases' ROI-based predictions; t is the grid value whose training-fold over-estimation rate is closest (ties: the less negative t). A Dice-optimal t (as in the voxel-probe recovery) is also reported as an extra arm, not used in the decision.
