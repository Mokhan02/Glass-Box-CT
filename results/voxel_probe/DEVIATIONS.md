# Deviations from PREREG.md

Each entry says what changed, why, and when relative to seeing any probe result.

## 1. Edge distance and tumour components treat unannotated voxels as unknown, not as non-tumour
- PREREG wording: "distance from the token centre to the nearest non-tumour voxel of the GT"; tumour components by 26-connectivity.
- Problem found in a 2-case smoke test of the extraction code (HCC_004, HCC_099), BEFORE any probe was fitted or any detection rate computed:
  for SEGs that annotate every other slice (the `geometry` route), voxels on unannotated slices are not tumour in the label map, so they act as
  a fake tumour edge: HCC_099 (a ~213 mm mass) had 0 voxels with edge distance > 2 mm and was split into 51 "components".
- Change: edge distance is the distance to the nearest KNOWN non-tumour voxel (unannotated voxels are treated like tumour for the distance
  computation); components are formed after bridging unannotated slabs along the slice axis (closing of length 7 voxels, restricted to
  tumour-or-unknown voxels), and counted by the tumour voxels they contain.
- Effect on the decision rule: none to its numbers or probes; it changes which tokens count as "interior" for the `geometry` route only.

## 2. Sensitivity analyses on clean cohorts, and a rule for disagreement (logged BEFORE any probe or decision result existed)
- Written while the extraction was still running; at that moment no `probe_table.csv` or `decision.json` existed on the instance (trained extraction 93/103, random-init not started).
- Finding that prompted it (see `results/exp1_audit/AUDIT.md` section 5): 22 of the 103 labelled HCC series contain more than one acquisition (distinct `AcquisitionNumber`), all in the `geometry` route;
  the converter interleaved their slices into one volume, which is not a valid CT image. Geometry-route labels also have unverified alignment (their SEGs reference images absent from the collection).
- **Primary analysis: unchanged.** The pre-registered run on all 103 cases remains the official result, with its numbers, probes and decision rule exactly as in PREREG.md.
- **Sensitivity analyses (added):** the entire pipeline (folds, probes, controls, head check, decision rule) is re-run, with its own patient folds and bootstrap resamples, on
  (a) the 81 single-acquisition cases and (b) the 75 UID-route cases. Nothing else is varied. Output folders: `results/voxel_probe/sensitivity_single_acquisition/` and `.../sensitivity_uid_route/`.
- **Rule for disagreement (fixed now):** if the primary verdict and the clean-cohort verdicts differ, **the clean cohorts govern interpretation**, because the 22 interleaved volumes are not valid CT images and a "miss" on one may be the model failing on scrambled input.
  Concretely: if (a) and (b) reach the same verdict, that verdict governs; if (a) and (b) reach different verdicts, the governing verdict is INCONCLUSIVE; the primary verdict is always reported alongside.
  `scripts/voxel_probe_governing.py` applies this mechanically.
- Code change: `scripts/voxel_probe_analyze.py` gains `--cohort {all,single_acquisition,uid_route}` (default `all` reproduces the primary analysis unchanged) and `--acq`. No change to probes, features, thresholds or the decision rule.
- Data fix, separate from the analysis: a corrected conversion (one acquisition per series, the one the segmentation references, unresolved cases excluded and logged) is being built in a NEW directory (`cache/heldout_nii_clean/`);
  the directory the pre-registered extraction reads is not modified.

## 3. The random-initialised control was invalid as implemented; corrected run added (found AFTER the primary results were visible)
- **What went wrong:** `voxel_probe_extract.py` computed each token's predicted-tumour coverage from the network being probed. For the randomly initialised network that prediction is meaningless, its "FN" group came out
  empty, and every random-init row in `probe_table.csv` is NaN (the decision file's `random_note` "random-init CI separate from trained" is therefore wrong: comparisons with NaN are false). The pre-registered
  rule did not use the random model (PREREG.md calls it an "interpretation note"), so **the primary verdict is unaffected**, but the control is unavailable in the primary output.
- **Timing and honesty:** noticed on reading the primary `probe_table.csv`, after the primary numbers and verdict were known. The fix is mechanical (it does not depend on any result): the control's token groups
  (tumour / liver, predicted / missed, interior) must be those of the TRAINED model on identical tokens, with only the activations replaced by the random network's.
- **Fix:** `scripts/voxel_probe_analyze.py --random-groups-from-trained` (default off, so every earlier run is reproduced unchanged). It asserts that the trained and random token tables have identical positions, copies the trained
  model's group fields onto the random tokens, and everything else (folds, probe, capping, threshold, bootstrap) is unchanged. Output: `results/voxel_probe/random_control_corrected/`. Reported as a SUPPLEMENTARY result; it does not
  replace or alter the primary decision.
- The same limitation applies to the sensitivity cohorts launched earlier: their random-init rows are also NaN, for the same reason.
