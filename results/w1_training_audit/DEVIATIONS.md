# W1 deviations and post-data decisions

Format: number, date, what, why, and whether any contrast value had been computed when it was decided.

## 1. CRML-CT: which segmentation series and which segments (2026-10-12)
PREREG says "the DICOM-SEG liver segment minus all tumour segments (checked on the first cases)". The metadata survey (`scripts/w1_crml_survey.py`, `crml_series_survey.csv`, metadata only, no pixel data, no contrast) showed each patient has 2 or 3 SEG series:
- `Segmentation`, SegmentAlgorithmType MANUAL, 197 patients: segments Liver, Liver Remnant, Hepatic (vein), Portal (vein), Tumor_1..Tumor_N.
- `AIMI liver-tumor AI segmentation`, AUTOMATIC, 197 patients: Liver, Mass. An algorithm output, not a label.
- `AIMI liver-tumor radiologist 7 corrected segmentation`, SEMIAUTOMATIC, 27 patients: an AI segmentation corrected by one reader.
Decision: the expert label is the series whose segments are all MANUAL; AUTOMATIC and SEMIAUTOMATIC series are never used (they are model output). Segment roles come from `SegmentLabel` (exact match: `Liver`, `Liver Remnant`, `Tumor_<n>`), not from segment order or from the type code meaning, because `Liver` and `Liver Remnant` both carry the code meaning "Liver" and would be conflated. Liver mask = segment `Liver` only (remnant, veins and other segments are not used); tumour = union of `Tumor_<n>`; tumour voxels are removed from the liver by the contrast function. One manual series per patient is required.
Contrast values computed when decided: none (one prototype patient was run to test the loader; its value is not used for any decision).

## 2. CRML-CT: patient exclusion criteria, fixed before any contrast is computed (2026-10-12)
Applied by `scripts/w1_crml_extract.py inventory`, which computes no contrast. A patient is excluded, with the reason logged, if: no single manual series; CT or SEG cannot be read; a SEG frame cannot be placed on a CT slice by its referenced SOP instance; no `Liver` or no `Tumor_<n>` segment, or either is empty; CT slice gaps deviate from the median by more than 1 mm; fewer than 50% of tumour voxels lie inside the `Liver` segment, or tumour volume exceeds liver volume (label inconsistency, the guard against a liver/tumour swap). Patients with no tumour of at least 10 mm are kept but contribute no lesions; they are listed separately.
Contrast values computed when decided: none.
