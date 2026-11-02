"""W4 scouting (CPU): WAW-TACE inventory from ct_hcc_metadata_v2.csv and the tumour masks (results/w4_waw/SCOUTING.md).
Per patient: which phases exist (0 native, 1 arterial, 2 portal venous, 3 delayed) and on which phase the tumour mask(s) were drawn; per tumour mask: connected components and the longest axial (Feret) diameter with the project's lesion function (glassbox.audit.contrast).
Usage (run with python -I): python -I scripts/w4_waw_inventory.py <raw dir (ct_hcc_metadata_v2.csv)> <extracted tumor mask dir> <out dir>"""
import collections
import csv
import glob
import json
import os
import re
import sys

import nrrd
import numpy as np

sys.path.insert(0, os.getcwd())
from glassbox.audit.contrast import select_lesions  # noqa: E402

raw, tdir, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
meta = list(csv.DictReader(open(os.path.join(raw, "ct_hcc_metadata_v2.csv"))))
P = collections.defaultdict(dict); thick = {}
for r in meta:
    P[r["patient_id"]][int(r["ct_phase"])] = int(r["tumor_count"]); thick[(r["patient_id"], int(r["ct_phase"]))] = float(r["slice_thickness"])
rows = []
for f in sorted(glob.glob(os.path.join(tdir, "*", "*", "*_tumor_seg.nrrd"))):
    m = re.match(r"(\d+)_(\d)_(\d+)_tumor_seg\.nrrd", os.path.basename(f)); pid, ph, idx = m.group(1), int(m.group(2)), int(m.group(3))
    d, h = nrrd.read(f); sp = np.abs(np.diag(h["space directions"])).astype(float); lab, sel = select_lesions(d > 0, tuple(sp), n_lesions=None, min_diam_mm=0.0)
    diam = [s[0] for s in sel]; vox = float(np.prod(sp))
    rows.append({"patient": pid, "phase": ph, "mask_index": idx, "components": len(sel), "largest_diam_mm": max(diam) if diam else 0.0, "volume_cm3": float((d > 0).sum() * vox / 1000), "all_diam_mm": ";".join(f"{x:.1f}" for x in diam),
                 "spacing_mm": "x".join(f"{x:.3f}" for x in sp)})
with open(os.path.join(out, "tumour_masks.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# consistency with the metadata (tumor_count per scan = number of mask files drawn on that scan)
cnt = collections.Counter((r["patient"], r["phase"]) for r in rows); mism = [(p, ph, P[p].get(ph), c) for (p, ph), c in cnt.items() if P[p].get(ph) != c]
S = {"patients": len(P), "scans_by_phase": dict(collections.Counter(ph for v in P.values() for ph in v)), "mask_files": len(rows), "metadata_mismatches": len(mism),
     "phase_set_by_patient": {str(k): v for k, v in collections.Counter(tuple(sorted(v)) for v in P.values()).items()},
     "masked_phase_by_patient": {str(k): v for k, v in collections.Counter(tuple(sorted({ph for ph, c in v.items() if c > 0})) for v in P.values()).items()},
     "mask_files_by_phase": dict(collections.Counter(r["phase"] for r in rows)),
     "mask_files_ge10mm_by_phase": dict(collections.Counter(r["phase"] for r in rows if r["largest_diam_mm"] >= 10)),
     "components_ge10mm_by_phase": {str(ph): sum(1 for r in rows if r["phase"] == ph for x in r["all_diam_mm"].split(";") if x and float(x) >= 10) for ph in range(4)}}
A = [p for p, v in P.items() if 1 in v and 2 in v]; Am = [p for p in A if any(c > 0 for c in P[p].values())]; A_arPV = [p for p in A if P[p].get(1, 0) > 0 or P[p].get(2, 0) > 0]
by_p = collections.defaultdict(list)
for r in rows:
    by_p[r["patient"]].append(r)
ge10 = lambda ps: sum(1 for p in ps for r in by_p[p] for x in r["all_diam_mm"].split(";") if x and float(x) >= 10)      # noqa: E731
S.update({"patients_with_arterial_and_portal_scans": len(A), "of_those_with_a_mask_on_phase_1_or_2": len(A_arPV), "of_those_with_any_mask": len(Am),
          "patients_arterial_AND_PV_scans_mask_on_arterial": sum(1 for p in A if P[p].get(1, 0) > 0), "patients_arterial_AND_PV_scans_mask_on_PV": sum(1 for p in A if P[p].get(2, 0) > 0),
          "patients_arterial_AND_PV_scans_mask_on_delayed_or_native_only": sum(1 for p in A if P[p].get(1, 0) == 0 and P[p].get(2, 0) == 0),
          "lesions_ge10mm_in_those_patients(masked_phase_1_or_2)": ge10(A_arPV), "lesions_ge10mm_all_patients": ge10(list(P)),
          "patients_with_ge1_lesion_ge10mm": sum(1 for p in P if any(float(x) >= 10 for r in by_p[p] for x in r["all_diam_mm"].split(";") if x))})
json.dump(S, open(os.path.join(out, "inventory.json"), "w"), indent=1); print(json.dumps(S, indent=1))
