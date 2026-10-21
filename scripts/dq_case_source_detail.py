"""Detailed source description of one HCC case (CPU only): every CT and SEG series in the study (description, slice count, z range, acquisition number), the manual SEG's referenced series, the z position of every SEG frame against the CT instance it references,
InstanceNumber order against z order in the CT series the converter used, and whether the case is a geometry-route / multi-acquisition case touched by the earlier converter fixes. Usage (cwd = cache root): python scripts/dq_case_source_detail.py <case dir name> <out json>"""
import csv
import glob
import json
import re
import sys

import numpy as np
import pydicom

case, out = sys.argv[1:3]; pid = re.match(r"^(HCC_\d+)", case).group(1); meta = json.load(open(f"heldout_nii/hcc_tace_seg/{case}/segments.json")); study = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/CT_{meta['ct_series'].replace('CT_', '')}")[0].rsplit("/", 1)[0]
R = {"case": case, "label_method": meta.get("label_method", "uid"), "segments": meta["segments"], "ct_series_used": meta["ct_series"], "seg_series_used": meta["seg_series"], "ct_series": [], "seg_series": []}
for d in sorted(glob.glob(study + "/CT_*")):
    hs = [pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(d + "/*.dcm")]; n = np.cross(*np.array(hs[0].ImageOrientationPatient, float).reshape(2, 3)); z = np.array([float(np.dot(np.array(h.ImagePositionPatient, float), n)) for h in hs])
    R["ct_series"].append({"series": d.rsplit("/", 1)[1][-8:], "description": str(hs[0].get("SeriesDescription", "")), "n": len(hs), "z_min": round(float(z.min()), 1), "z_max": round(float(z.max()), 1), "acquisition": str(hs[0].get("AcquisitionNumber", "")), "slice_thickness": str(hs[0].get("SliceThickness", "")), "used": d.endswith(meta["ct_series"].replace("CT_", ""))})
for d in sorted(glob.glob(study + "/SEG_*")):
    h = pydicom.dcmread(sorted(glob.glob(d + "/*.dcm"))[0], stop_before_pixels=True); R["seg_series"].append({"series": d.rsplit("/", 1)[1][-8:], "description": str(h.get("SeriesDescription", "")), "n_frames": int(h.get("NumberOfFrames", 0)), "used": d.endswith(meta["seg_series"].replace("SEG_", ""))})
cd = glob.glob(study + "/CT_" + meta["ct_series"].replace("CT_", ""))[0]; hs = {str(h.SOPInstanceUID): h for h in (pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(cd + "/*.dcm"))}; n = np.cross(*np.array(next(iter(hs.values())).ImageOrientationPatient, float).reshape(2, 3))
order_z = sorted(hs.values(), key=lambda h: float(np.dot(np.array(h.ImagePositionPatient, float), n))); inst = [int(h.InstanceNumber) for h in order_z]
R["instance_number_vs_z"] = {"instance_numbers_in_z_order_first10": inst[:10], "monotonic_increasing": bool(np.all(np.diff(inst) > 0)), "monotonic_decreasing": bool(np.all(np.diff(inst) < 0))}
seg = pydicom.dcmread(glob.glob(study + "/SEG_" + meta["seg_series"].replace("SEG_", "") + "/*.dcm")[0], stop_before_pixels=True); zi = {id(h): i for i, h in enumerate(order_z)}; fr = []
for f in seg.PerFrameFunctionalGroupsSequence:
    s = str(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID); h = hs.get(s)
    if h is not None:
        fr.append((zi[id(h)], float(np.dot(np.array(f.PlanePositionSequence[0].ImagePositionPatient, float), n)), float(np.dot(np.array(h.ImagePositionPatient, float), n))))
fr = np.array(fr); R["seg_frames"] = {"n": len(fr), "ct_slice_index_range": [int(fr[:, 0].min()), int(fr[:, 0].max())], "max_abs_z_difference_seg_vs_referenced_ct_mm": round(float(np.abs(fr[:, 1] - fr[:, 2]).max()), 3), "n_ct_slices": len(order_z)}
acq = [r for r in csv.DictReader(open("hcc_acq_check.csv")) if r["case"] == case]; R["hcc_acq_check_row"] = acq[0] if acq else None
json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
