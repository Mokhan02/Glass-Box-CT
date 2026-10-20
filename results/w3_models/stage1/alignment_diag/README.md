# Alignment diagnostic for the 3 cases that failed the model A organ-Dice check (no tumour metric computed)

Script `scripts/w3_alignment_diag.py` (verdict rule fixed in the script docstring before the run). Per-case numbers: `w3_alignment_diagnostic.json`; overlays: `overlay_*.png` (green = GT liver, red = predicted liver, cyan = spleen, yellow = right kidney, orange = left kidney).

| case | liver Dice | centroid offset | bbox IoU | best shift Dice | verdict (rule) | what the overlay shows |
|---|---|---|---|---|---|---|
| HCC_019_773068 | 0.654 | 24.2 mm (all in y) | 0.768 | 0.925 after a 36-voxel (25 mm) y shift of the prediction | MISALIGNED | the predicted liver sits on the CT liver; the **ground-truth label is displaced ~25 mm posteriorly relative to the CT** |
| HCC_022_228695 | 0.830 | 13.4 mm (z) | 0.991 | 0.830 (no shift helps) | ALIGNED-BUT-UNDERSEGMENTED | very large tumour: the prediction misses part of the tumour-bearing liver (recall 0.74) |
| HCC_043_848214 | 0.836 | 9.8 mm | 0.861 | 0.839 | ALIGNED-BUT-UNDERSEGMENTED | prediction extends into adjacent bowel / stomach (precision 0.85) and misses part of the left lobe |

In all three, output shape equals the input shape, the CT affine axcodes are LPS, spleen and kidneys sit where they should relative to the liver (spleen left of the liver, right kidney right of the left kidney, both kidneys inferior to the liver centroid) and their volumes are plausible (spleen 120 to 582 mL, kidneys 117 to 193 mL). No GT exists for spleen or kidney in HCC-TACE-Seg, so these are sanity checks, not Dice.
