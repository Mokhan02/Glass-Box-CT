# Data-quality report: HCC_019_773068 ground-truth label displaced 25 mm (2026-10-20)

All checks CPU only (the GPU was busy training). Code: `scripts/dq_source_check.py`, `dq_label_offset.py`, `dq_screen_report.py`, `dq_reconvert_seg_offset.py`, `dq_impact_case.py`, `dq_ref_corrected.py`, `dq_impact.py`, `dq_overlay.py`.

## 1. Source check for HCC_019_773068: the displacement is in how our converter read the SEG, not in a wrong source label
- The manual SEG (series description "Segmentation", 268 frames) references the same CT series we used (same SeriesInstanceUID) and has the same FrameOfReferenceUID, Rows/Columns, PixelSpacing and ImageOrientationPatient.
- **Every SEG frame's ImagePositionPatient is offset by exactly 25.0 mm along the row-index direction (array y) from the CT instance it references by SOPInstanceUID** (0.0 mm along x and through-plane). The SEG declares its own position 25 mm away from the CT it references.
- `scripts/convert_heldout.py` (UID route) reads the SEG with `get_pixels_by_source_instance(..., ignore_spatial_locations=True)`, i.e. it discards the SEG's declared positions and places each frame by index. That is how a 25 mm (35.6 voxel) displacement entered our converted label.
- Honouring the SEG's own metadata fixes it: shifting the label by the declared offset (36 voxels, rounding error at most 0.35 mm) raises liver Dice against model A's liver mask from 0.654 to 0.925 (35 voxels: 0.926). The model-free edge screen independently estimates the displacement at -26.7 mm in y.
- Not touched by the earlier converter fixes: it is a UID-route case (`label_method` uid, 268 of 268 frames matched), not one of the geometry-route cases in `results/exp1_audit/hcc_clean_conversion.csv`; its liver label was converted once, as published.
- **Classification (rule 3): converter bug**, not a source-label error: the source SEG's own spatial metadata places the label correctly.

## 2. Cohort screen of the 89 clean cases (no tumour metrics): `SCREEN.md`, `screen_table.csv`
- **Source check on all 89** (SEG header vs the CT series used): the SEG's referenced series equals the CT used in 89 of 89; in-plane SEG-CT position offset above 1 mm in **1 case (HCC_019_773068)**; through-plane offsets at most 1.25 mm (under half a slice); FrameOfReferenceUID differs in 1 case (HCC_012_311550) with identical positions.
- **Limitation, stated plainly:** no VISTA-3D liver MASKS were ever stored (only the tumour masks and the per-case liver Dice), so the requested VISTA liver centroid offset / bounding-box IoU cannot be computed from stored data without a GPU run. Two substitutes were used: the **stored VISTA liver Dice** (hackathon evaluation, annotated slices, matched to each case's conversion) and a **model-free edge-alignment screen** (the GT liver boundary shifted over +-40 mm in-plane and +-3 slices; the shift maximising mean CT gradient magnitude is the estimated label displacement).
- Flags: stored VISTA liver Dice < 0.80 in 3 cases (HCC_099_597879 0.511, HCC_019 0.646, HCC_022_228695 0.793); edge screen (shift > 10 mm and gain > 25%, threshold fixed before the cohort run) in 3 cases (HCC_012, HCC_019, HCC_024).
- **Visual review of every flagged case** (`overlays/`): HCC_019 is displaced (green GT off the liver by ~25 mm). **HCC_012 and HCC_024 are edge-screen false positives** (the GT contour follows the liver; the estimated shifts of 33 to 39 mm sit at the edge of the search range and land on other organs). HCC_099 and HCC_022 are low-Dice from model failure on very large tumour-bearing livers (the GT contour is accurate in HCC_099; model A also under-segments HCC_022). **Only HCC_019 is a displaced label.**

## 3. Rule applied (fixed by the researcher before the check): converter bug -> fix, re-convert, re-run the affected cases, no exclusion
- Corrected re-conversion `heldout_nii_fixed/hcc_tace_seg/HCC_019_773068` (instance, persistent filesystem) by `scripts/dq_reconvert_seg_offset.py`: the original conversion is reproduced voxel for voxel, then shifted by the SEG-declared offset; CT unchanged; the original conversion is left untouched (it is the "as converted" sensitivity). Voxel counts per segment are unchanged by the shift (liver 1,361,178, tumour 54,407).
- **The lesion changes group.** The single cohort lesion (58.9 mm) had contrast -49.2 HU (dark) in the displaced label and has **+8.2 HU (bright)** in the corrected one. Under the stored VISTA-3D prediction it is a miss either way (coverage 0.0022 displaced, 0.0019 corrected).
- Re-run of the affected cases: the hackathon quantities (RECIST, detection, W1 contrast, the W3 reference row) depend only on the stored prediction and the label and are recomputed here. **Still to do, needs the GPU after training:** the point-prompted W2 v3 HCC arms (`hcc_26_3`, `hcc_133_1`) place the click at the interior point of the (displaced) lesion, so that lesion's 2 rows must be re-run with the corrected label, together with any W2-M step that uses its group; nothing is re-run now. The W3 cohort in PREREG_stage1 (28 bright / 58 dark) would become 29 / 57 with the corrected case; that change is NOT applied (see the proposal in the log).

## 4. Impact (`dq_impact.md`, `dq_impact.json`), three scenarios: as converted (published), case EXCLUDED, case CORRECTED
| quantity | published | excluded | corrected |
|---|---|---|---|
| UID-route sum-of-diameters bias | -19.4% [-28.1, -12.8] | -19.0% [-26.9, -12.8] (moves 0.4 pt) | -19.4% (no change: still a miss) |
| clean-89 bias | -21.1% [-30.4, -14.5] | -20.4% [-29.3, -13.6] (moves 0.7 pt) | -21.1% (no change) |
| UID-route detection | 70/86 = 81.4% | 70/85 = 82.4% | 70/86 = 81.4% |
| W1 HCC bright share | 32.6% [21.2, 42.4] (28/86); BRIGHT RARE IN TRAINING | 32.9% [22.3, 43.3] (28/85); same verdict | 33.7% [22.2, 43.5] (29/86); same verdict |
| W2 v3 HCC Dice gaps (excluded only) | 26/3 +29.0 GAP; 133/1 +24.4 GAP | 26/3 +30.1 GAP; 133/1 +25.2 GAP | needs GPU re-run (2 rows) |
| W3 reference row (VISTA 26 auto) | detection +41.3 GAP, Dice +39.9 GAP | +42.9 GAP, +41.1 GAP | 29 bright / 57 dark: +44.8 GAP, +41.9 GAP |

**Pre-committed rule:** the bias moves by more than 2 points or any verdict changes category -> update the hackathon Evidence page; otherwise a RESEARCH_LOG footnote. **Result: the headline bias moves by at most 0.7 point and no verdict changes category in any computed scenario, so this is a RESEARCH_LOG footnote.** Provisional on one item: the corrected-label W2 point-prompt arms are not yet re-run; if that re-run changes a W2 v3 verdict category the rule triggers and I will say so.
