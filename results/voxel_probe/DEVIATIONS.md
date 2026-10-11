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
