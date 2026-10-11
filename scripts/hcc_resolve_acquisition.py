"""For each geometry-route HCC case, find which CT instances the manual SEG actually references, by SOP instance UID, across ALL CT series of the study
(not just the series my converter picked). Reports, per case, the series and AcquisitionNumber holding the referenced instances, and how many referenced instances exist at all.
Usage: python scripts/hcc_resolve_acquisition.py cache/hcc_resolve.csv"""
import collections
import csv
import glob
import json
import os
import re
import sys

import pydicom

root, held = "cache/heldout_nii/hcc_tace_seg", "cache/heldout/hcc_tace_seg"
rows = []
for case in sorted(os.listdir(root)):
    meta = json.load(open(os.path.join(root, case, "segments.json")))
    if meta.get("label_method", "uid") != "geometry":
        continue
    pid = re.match(r"^(HCC_\d+)", case).group(1)
    seg_dir = glob.glob(f"{held}/{pid}/*/{meta['seg_series']}")[0]; study = os.path.dirname(seg_dir)
    seg = pydicom.dcmread(glob.glob(seg_dir + "/*.dcm")[0])
    ref = set()
    for f in seg.PerFrameFunctionalGroupsSequence:
        for d in f.get("DerivationImageSequence", []) or []:
            for s in d.get("SourceImageSequence", []) or []:
                ref.add(str(s.ReferencedSOPInstanceUID))
    for sq in seg.get("ReferencedSeriesSequence", []) or []:
        for i in sq.get("ReferencedInstanceSequence", []) or []:
            ref.add(str(i.ReferencedSOPInstanceUID))
    where = collections.Counter(); per_series = collections.defaultdict(lambda: collections.Counter())
    n_files = 0
    for d in glob.glob(f"{study}/CT_*"):
        for fp in glob.glob(d + "/*.dcm"):
            h = pydicom.dcmread(fp, stop_before_pixels=True); n_files += 1
            if str(h.SOPInstanceUID) in ref:
                where[(os.path.basename(d)[3:][-8:], str(h.get("AcquisitionNumber", "")))] += 1
    found = sum(where.values()); best = where.most_common(2)
    rows.append({"case": case, "n_frames_ref_sops": len(ref), "ref_sops_found_in_study": found, "found_frac": round(found / max(len(ref), 1), 3), "used_series_suffix": meta["ct_series"][-8:],
                 "best_series_acq": f"{best[0][0][0]}/acq{best[0][0][1]}:{best[0][1]}" if best else "", "second": f"{best[1][0][0]}/acq{best[1][0][1]}:{best[1][1]}" if len(best) > 1 else "", "n_ct_files_in_study": n_files})
    print(rows[-1], flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} geometry cases; referenced instances found in the study for >=95%: {sum(r['found_frac'] >= 0.95 for r in rows)}; for 0%: {sum(r['found_frac'] == 0 for r in rows)}; partial: {sum(0 < r['found_frac'] < 0.95 for r in rows)}")
