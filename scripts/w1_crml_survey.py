"""W1: metadata-only survey of the CRML-CT DICOM-SEG series (no pixel data, no contrast). Writes one row per series with its segment labels and code meanings.
Usage: python -I scripts/w1_crml_survey.py data/w1/crlm results/w1_training_audit/crml_series_survey.csv"""
import csv
import glob
import os
import sys

import pydicom

root, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(os.listdir(root)):
    for sdir in sorted(glob.glob(os.path.join(root, p, "SEG_*"))):
        fs = glob.glob(os.path.join(sdir, "*.dcm"))
        d = pydicom.dcmread(fs[0], stop_before_pixels=True)
        segs = [(int(s.SegmentNumber), str(s.SegmentLabel), s.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning, s.SegmentedPropertyTypeCodeSequence[0].CodeMeaning,
                 str(s.get("SegmentAlgorithmType", ""))) for s in d.SegmentSequence]
        rows.append({"patient": p, "series": os.path.basename(sdir), "n_files": len(fs), "description": str(d.get("SeriesDescription", "")), "n_frames": int(d.get("NumberOfFrames", 0)),
                     "algorithm_types": "|".join(sorted({s[4] for s in segs})), "segments": " ; ".join(f"{n}:{l} [{t}]" for n, l, c, t, a in segs),
                     "n_tumour_like": sum(l.lower().startswith("tumor") or l.lower().startswith("tumour") for n, l, c, t, a in segs),
                     "device": str(d.get("ManufacturerModelName", "")), "content_creator": str(d.get("ContentCreatorName", ""))})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "series,", len({r["patient"] for r in rows}), "patients")
