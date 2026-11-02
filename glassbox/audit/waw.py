"""WAW-TACE reference parenchyma for the tumour-minus-liver contrast (W4 scouting, results/w4_waw/CONTRAST_RULES.md).
The dataset's organ masks are TotalSegmentator outputs of its 117-class `total` task (official class map, `totalsegmentator/map_to_binary.py`): 5 = liver, 64 = portal_vein_and_splenic_vein (63 = inferior_vena_cava).
Parity with the HCC exp8 reference (label 1 = liver parenchyma with every tumour component and the labelled portal vein removed, hepatic veins left in): reference = liver (label 5) minus ALL tumour voxels of the scan minus the portal/splenic vein (label 64).
The organ mask is single-valued, so label 64 never overlaps label 5 and the exclusion is a guarantee, not a change, for these files (checked on every mask in results/w4_waw/mask_check_cohort.json); it is applied explicitly so the rule does not depend on that property.
Hepatic veins have no TotalSegmentator class: they stay inside label 5, as in exp8."""
import numpy as np

LIVER, PORTAL_SPLENIC_VEIN = 5, 64


def reference_parenchyma(organ, tumour_all):
    """organ: integer label array; tumour_all: bool array of every tumour voxel of the same scan (all components). Returns the bool reference array."""
    organ = np.asarray(organ)
    return (organ == LIVER) & ~np.asarray(tumour_all, bool) & (organ != PORTAL_SPLENIC_VEIN)
