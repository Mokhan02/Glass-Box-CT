# HCC_027_967743: source-label displacement (2026-10-20); excluded under the pre-committed rule

CPU only, apart from 15 minutes of GPU to obtain model A and model D liver masks for the 14 position-matched cases that had never been run (needed for the 89-case re-screen).

## 1. Source check: not a converter / slice-mapping bug
- UID-route case (`label_method` uid, 316 of 316 SEG frames matched their referenced CT instance); not a geometry-route or multi-acquisition-resolved case and not touched by the earlier converter fixes (`hcc_clean_conversion.csv` lists geometry cases only; `hcc_acq_check.csv`: one acquisition number in the used series, regular 2.5 mm spacing).
- The manual SEG ("Segmentation", 316 frames = 79 slices x 4 segments) references the CT series we used (acquisition 3, 79 slices, 2.5 mm). The z position of every SEG frame equals the z position of the CT instance it references (maximum difference 0.0 mm); in-plane offsets 0.0 mm; same FrameOfReferenceUID. The CT series sorts by z with InstanceNumber descending (79 to 1), which is irrelevant for a UID-mapped read. `source_detail_HCC_027.json`.
- The study holds three CT series: a 5 mm "PRE LIVER" series, the 2.5 mm acquisition-3 series we used, and a second 2.5 mm series (acquisition 1) with the identical z grid. **The two 2.5 mm series are anatomically offset by 4 slices (10 mm):** the normalised cross-correlation of the volumes peaks at a 4-slice shift (0.805) against 0.758 unshifted (`series_shift_HCC_027.json`, `scripts/dq_series_shift.py`).
- **The label matches the anatomy of the other acquisition, in the direction that fits:** both model A and model D place the liver on slices 18 to 78 of the used series while the GT liver spans 13 to 74, and the best translation of either model's liver onto the GT is -10.0 mm (4 slices) through-plane (organ Dice 0.854 to 0.947 for A, 0.854 to 0.948 for D). This is consistent with the annotator having drawn on the acquisition-1 series while the SEG records the acquisition-3 instance UIDs. It is consistent with, not proof of, that explanation.
- **Classification: source annotation error (not a converter bug).** Our reader places each frame where the SEG declares it, exactly.

## 2. 89-case re-screen (`rescreen_translation.json`, `scripts/dq_rescreen_translation.py`)
Dice-maximising translation test over +-40 voxels in every axis (through-plane included) with BOTH models' predicted liver; flag = best shift > 5 mm on some axis AND organ Dice gain > 0.05. Model A and D masks for the 75 UID cases came from the Stage 1 runs; for the 14 position-matched cases both models were run once for this screen. **Result: only HCC_027_967743 is flagged, by both models, with the same shift (A: [0.7, 0.7, -10.0] mm, gain 0.093; D: [0.7, 0.7, -10.0] mm, gain 0.094). The two models agree on every case (no case is flagged by only one).** HCC_019 (corrected) is aligned. No other case has a best shift above 5 mm on any axis. Limit: the test needs a reasonable liver prediction; a liver that is wrong by a different mechanism would not show a translation.

## 3. Rule applied (fixed by the researcher): converter bug -> fix; source error -> exclude from ALL cohorts, with a sensitivity including them
HCC_027 is a source error, so it is **excluded from the primary analysis of every cohort**, with the including-version reported as a sensitivity. Applied in W3 code (`results/dq_hcc_labels/excluded_cases.json`; W3 cohort now 85 lesions / 74 cases, 28 bright / 57 dark; PREREG Amendment 3). The published W1, W2 and hackathon-reference tables are frozen; their excluded-case numbers are recomputed in `dq_impact_027.md` and are within the pre-committed bounds below (so the hackathon Evidence page is not updated).

## 4. Impact (`dq_impact_027.md`, baseline = HCC_019 corrected, HCC_027 included)
| quantity | included | excluded |
|---|---|---|
| UID-route sum-of-diameters bias | -19.4% [-28.1, -12.8] | -19.0% [-28.1, -12.8] (+0.46 pt) |
| clean-89 bias | -21.1% [-30.4, -14.5] | -20.4% [-29.0, -13.6] (+0.66 pt) |
| UID detection | 70/86 = 81.4% | 69/85 = 81.2% |
| W1 HCC bright share | 33.7% [22.2, 43.5] (29/86) | 32.9% [22.2, 43.2] (28/85); both BRIGHT RARE IN TRAINING |
| W2 v3 HCC Dice gaps | 26/3 +28.6, 133/1 +23.8, both GAP | +27.9 and +22.9, both GAP; kidney unchanged |
| W3 (A, D, VISTA reference) | all GAP (detection and Dice), 29/57 | all GAP (detection and Dice), 28/57 |
**Rule (bias moves > 2 pts or any verdict category changes): not triggered** (largest bias move 0.66 point; no W1, W2 or W3 verdict changes).
