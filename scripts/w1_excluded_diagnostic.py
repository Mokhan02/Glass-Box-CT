"""W1 diagnostic (NOT used in any analysis): why did each MSD Task03 case not enter, and is the reason related to tumour size or brightness?
Reads msd_inventory.csv and msd_lesions.csv; for the cases that failed the DEV5 surface guard it also computes lesion contrast with the shared function (liver = label 1) so brightness can be compared.
Usage: python scripts/w1_excluded_diagnostic.py <Task03_Liver dir> results/w1_training_audit"""
import csv
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402

root, out = sys.argv[1:3]
inv = list(csv.DictReader(open(f"{out}/msd_inventory.csv"))); les = list(csv.DictReader(open(f"{out}/msd_lesions.csv")))
inc = [r for r in inv if r["included"] == "True"]
tv = np.array([float(r["tumour_vox"]) for r in inc])
lines = ["# MSD Task03: the 16 cases that did not enter (diagnostic only, not used in any analysis)", "",
         f"Included cases: {len(inc)}; tumour voxels, median {np.median(tv):.0f}, 5th-95th percentile {np.percentile(tv, 5):.0f} to {np.percentile(tv, 95):.0f}.", ""]
tf = [r for r in inv if r["reason"].startswith("tumour-free")]
lines += [f"- {len(tf)} tumour-free cases (no label 2): " + ", ".join(r["case"] for r in tf) + ". They contain no tumour, so their exclusion cannot depend on tumour size or brightness (no tumour exists to measure).", ""]
for r in [r for r in inv if r["reason"].startswith("label inconsistency")]:
    name = r["case"] + ".nii.gz"
    img = nib.as_closest_canonical(nib.load(f"{root}/imagesTr/{name}")); lab = nib.as_closest_canonical(nib.load(f"{root}/labelsTr/{name}"))
    sp = tuple(float(z) for z in img.header.get_zooms()[:3]); ct = np.asarray(img.dataobj).astype(np.float32); L = np.asarray(lab.dataobj).astype(np.uint8)
    rows = case_lesions(ct, L == 2, L == 1, sp, n_lesions=None)
    pct = 100.0 * (tv < float(r["tumour_vox"])).mean()
    lines.append(f"- {r['case']}: failed the surface guard (tumour surface within 3 mm of liver label: {float(r['tumour_surface_near_liver']):.2f}; tumour/liver volume {float(r['tumour_over_liver_vol']):.3f}). "
                 f"Tumour voxels {int(float(r['tumour_vox']))} (percentile among included cases: {pct:.0f}); lesions >= 10 mm: {len(rows)}"
                 + ("; contrast (HU, diameter mm): " + ", ".join(f"{x['contrast_hu']:+.1f} ({x['diam_mm']:.0f})" for x in rows) if rows else "") + ".")
nol = sorted({r["case"] for r in inc} - {r["patient"] for r in les})
lines += ["", f"Included cases that contributed no lesion (no tumour component >= 10 mm): {', '.join(nol) if nol else 'none'}; this is the pre-registered 10 mm lesion rule, which is size-related by definition."]
open(f"{out}/msd_excluded_diagnostic.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
