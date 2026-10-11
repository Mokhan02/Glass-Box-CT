"""Which HCC-TACE-Seg scans carry the manual masks: baseline (pre-treatment) or follow-up (post-TACE)?
Reads DICOM headers of each converted case's CT series (study date and descriptions) and orders each patient's studies by date.
Usage: python scripts/hcc_scan_provenance.py"""
import collections
import glob
import json
import os
import re

import pydicom

root, heldout = "cache/heldout_nii/hcc_tace_seg", "cache/heldout/hcc_tace_seg"
recs = []
for case in sorted(os.listdir(root)):
    if not os.path.exists(os.path.join(root, case, ".done")):
        continue
    meta = json.load(open(os.path.join(root, case, "segments.json")))
    pid = re.match(r"^(HCC_\d+)", case).group(1)
    hit = glob.glob(f"{heldout}/{pid}/*/{meta['ct_series']}/*.dcm")
    ds = pydicom.dcmread(hit[0], stop_before_pixels=True)
    recs.append({"case": case, "patient": pid, "date": str(ds.get("StudyDate", "")), "study_desc": str(ds.get("StudyDescription", "")),
                 "series_desc": str(ds.get("SeriesDescription", "")), "contrast": str(ds.get("ContrastBolusAgent", "") or "-"), "kvp": str(ds.get("KVP", ""))})
by = collections.defaultdict(list)
for r in recs:
    by[r["patient"]].append(r)
first = later = single = 0
for p, rs in by.items():
    rs.sort(key=lambda r: r["date"])
    for i, r in enumerate(rs):
        r["rank"] = i
    if len(rs) == 1:
        single += 1
    first += 1
    later += len(rs) - 1
print(f"{len(recs)} cases from {len(by)} patients; patients with 1 labelled study: {single}; with >1: {len(by) - single}")
print(f"cases that are a patient's earliest labelled study: {sum(r['rank'] == 0 for r in recs)}; later studies: {sum(r['rank'] > 0 for r in recs)}")
dated = [r for r in recs if r["date"] and not r["date"].startswith("1970")]
print(f"cases with a usable StudyDate: {len(dated)}/{len(recs)}  (IDC may shift dates; only the ordering is meaningful)")
print("\nstudy descriptions:", collections.Counter(r["study_desc"] for r in recs).most_common(6))
print("series descriptions of the labelled CT:", collections.Counter(r["series_desc"] for r in recs).most_common(8))
print("contrast agent field:", collections.Counter(r["contrast"] for r in recs).most_common(4))
multi = [(p, [(r["case"].split("_", 2)[2], r["date"], r["series_desc"]) for r in rs]) for p, rs in by.items() if len(rs) > 1][:5]
print("\nexamples of patients with more than one labelled study (case suffix, date, series):")
for p, rs in multi:
    print(" ", p, rs)
