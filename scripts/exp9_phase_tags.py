"""Can contrast phase be labelled for the 89-case clean cohort from DICOM headers? Reads, for the labelled CT series of each case (and the other CT series in its study),
every phase-related field and reports how often each is present and informative. Exploratory.
Usage (instance): python scripts/exp9_phase_tags.py cache/phase_tags.csv"""
import collections
import csv
import glob
import json
import os
import re
import sys

import pydicom

FIELDS = ["ContrastBolusAgent", "ContrastBolusStartTime", "ContrastBolusStopTime", "ContrastBolusVolume", "ContrastBolusRoute", "ContrastFlowRate", "ContrastBolusIngredient",
          "AcquisitionTime", "SeriesTime", "ContentTime", "TriggerTime", "SeriesDescription", "StudyDescription", "ProtocolName", "ImageType", "AcquisitionNumber", "SeriesNumber"]
TOK = re.compile(r"(ARTERIAL|ART\b|LATE ART|PORTAL|VENOUS|\bPV\b|DELAY|EQUILIB|\bNON\s?-?CONTRAST|\bPRE\b|UNENHANCED|\bW/O\b|\bWO\b|\bW/\s|PHASE|\bAP\b|\bC/A/P|\bCAP\b)", re.I)
acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}; conv = {r["case"]: r["status"] for r in csv.DictReader(open("results/exp1_audit/hcc_clean_conversion.csv"))}
cohort = [(c, "uid") for c in sorted(acq) if acq[c]["route"] == "uid"] + [(c, "resolved") for c in sorted(conv) if conv[c] == "RESOLVED"]
held = "cache/heldout/hcc_tace_seg"; rows = []
for case, src in cohort:
    pid = re.match(r"^(HCC_\d+)", case).group(1)
    meta = json.load(open(os.path.join("cache/heldout_nii" if src == "uid" else "cache/heldout_nii_clean", "hcc_tace_seg", case, "segments.json")))
    cts = meta["ct_series"][3:] if meta["ct_series"].startswith("CT_") else meta["ct_series"]  # the two converters store it with / without the CT_ prefix
    sdir = glob.glob(f"{held}/{pid}/*/CT_{cts}")[0]; study = os.path.dirname(sdir)
    h = pydicom.dcmread(sorted(glob.glob(sdir + "/*.dcm"))[0], stop_before_pixels=True)
    row = {"case": case, "source": src}
    for f in FIELDS:
        v = h.get(f, ""); row[f] = "|".join(map(str, v)) if isinstance(v, (list, pydicom.multival.MultiValue)) else str(v)
    others = []
    for d in glob.glob(f"{study}/CT_*"):
        o = pydicom.dcmread(sorted(glob.glob(d + "/*.dcm"))[0], stop_before_pixels=True); others.append((str(o.get("AcquisitionTime", "")).split(".")[0], str(o.get("SeriesDescription", "")), os.path.basename(d)[3:] == cts))
    others.sort()
    row["n_ct_series_in_study"] = len(others); row["labelled_series_rank_by_acq_time"] = next(i for i, o in enumerate(others) if o[2]) + 1
    row["other_series_descriptions"] = " || ".join(o[1] for o in others if not o[2])[:200]; row["desc_tokens"] = ",".join(sorted({m.group(0).upper().strip() for m in TOK.finditer(row["SeriesDescription"] + " " + row["StudyDescription"])}))
    rows.append(row); print(case, row["SeriesDescription"][:38], "| agent:", row["ContrastBolusAgent"][:12] or "-", "| start:", row["ContrastBolusStartTime"] or "-", "| rank", row["labelled_series_rank_by_acq_time"], "/", row["n_ct_series_in_study"], flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = len(rows); print(f"\n{n} cases. Presence of each phase-related field (non-empty) in the labelled series:")
for f in FIELDS:
    present = sum(1 for r in rows if r[f] not in ("", "None")); distinct = len({r[f] for r in rows})
    print(f"  {f:26s} {present:3d}/{n}  ({distinct} distinct values)")
print("description tokens:", collections.Counter(t for r in rows for t in r["desc_tokens"].split(",") if t).most_common(12))
print("number of CT series in the study:", dict(collections.Counter(r["n_ct_series_in_study"] for r in rows)))
print("top series descriptions:", collections.Counter(r["SeriesDescription"] for r in rows).most_common(8))
