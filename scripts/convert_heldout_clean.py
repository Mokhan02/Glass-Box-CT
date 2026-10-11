"""Corrected conversion of the geometry-route HCC cases into a NEW directory (cache/heldout_nii_clean/hcc_tace_seg); the directory read by the
pre-registered voxel-probe extraction is not touched.
For each case: (1) read every SOP instance UID the manual SEG references; (2) take the CT series (anywhere in the study) holding most of them;
(3) inside it, an ACQUISITION (AcquisitionNumber) 'matches' if >= 95% of the SEG frames' 3D positions coincide (< 0.5 mm) with one of its instances;
(4) exactly one matching acquisition -> RESOLVED, volume built from that acquisition only; several matching (identical positions) -> AMBIGUOUS, a volume is written
per candidate (suffix _acq<k>) and none counts as clean; no match, no referenced instance, or any inconsistency -> UNRESOLVED, excluded, reason logged.
Usage: python scripts/convert_heldout_clean.py results/exp1_audit/hcc_clean_conversion.csv"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pydicom
import SimpleITK as sitk

held, old, out = "cache/heldout/hcc_tace_seg", "cache/heldout_nii/hcc_tace_seg", "cache/heldout_nii_clean/hcc_tace_seg"
log = []


def write_volume(case_dir, insts, frames, fpos, fseg, segs, info):
    insts = sorted(insts, key=lambda i: i["z"])
    zs = np.array([i["z"] for i in insts])
    if len(set(np.round(zs, 2))) != len(zs):
        raise RuntimeError("duplicate slice positions inside the chosen acquisition")
    reader = sitk.ImageSeriesReader(); reader.SetFileNames([i["file"] for i in insts]); ct = reader.Execute()
    label = np.zeros((len(insts), frames.shape[1], frames.shape[2]), np.uint8); matched = 0
    P = np.array([i["pos"] for i in insts])
    for k in range(len(fpos)):
        d = np.linalg.norm(P - fpos[k], axis=1); j = int(d.argmin())
        if d[j] < 0.5:
            label[j][frames[k] > 0] = fseg[k]; matched += 1
    if matched / len(fpos) < 0.95:
        raise RuntimeError(f"only {matched}/{len(fpos)} SEG frames placed")
    lab = sitk.GetImageFromArray(label); lab.CopyInformation(ct); os.makedirs(case_dir, exist_ok=True)
    sitk.WriteImage(ct, os.path.join(case_dir, "ct.nii.gz")); sitk.WriteImage(lab, os.path.join(case_dir, "seg.nii.gz"))
    json.dump({"segments": {str(n): l for n, l in segs}, "label_method": "sop_series+position", "n_slices": len(insts), "frames_matched": matched / len(fpos), **info}, open(os.path.join(case_dir, "segments.json"), "w"))
    open(os.path.join(case_dir, ".done"), "w").close()


for case in sorted(os.listdir(old)):
    meta = json.load(open(os.path.join(old, case, "segments.json")))
    if meta.get("label_method", "uid") != "geometry":
        continue
    row = {"case": case, "status": "", "detail": "", "used_by_old_converter": meta["ct_series"][-8:], "chosen_series": "", "candidates": ""}
    try:
        pid = re.match(r"^(HCC_\d+)", case).group(1); seg_dir = glob.glob(f"{held}/{pid}/*/{meta['seg_series']}")[0]; study = os.path.dirname(seg_dir)
        seg = pydicom.dcmread(glob.glob(seg_dir + "/*.dcm")[0]); frames = seg.pixel_array
        frames = frames[None] if frames.ndim == 2 else frames
        fpos = np.array([np.array(f.PlanePositionSequence[0].ImagePositionPatient, float) for f in seg.PerFrameFunctionalGroupsSequence])
        fseg = np.array([f.SegmentIdentificationSequence[0].ReferencedSegmentNumber for f in seg.PerFrameFunctionalGroupsSequence])
        segs = [(s.SegmentNumber, s.SegmentLabel) for s in seg.SegmentSequence]
        ref = set()
        for f in seg.PerFrameFunctionalGroupsSequence:
            for d in f.get("DerivationImageSequence", []) or []:
                for s in d.get("SourceImageSequence", []) or []:
                    ref.add(str(s.ReferencedSOPInstanceUID))
        for sq in seg.get("ReferencedSeriesSequence", []) or []:
            for i in sq.get("ReferencedInstanceSequence", []) or []:
                ref.add(str(i.ReferencedSOPInstanceUID))
        insts, per_series = [], collections.Counter()
        for d in glob.glob(f"{study}/CT_*"):
            sname = os.path.basename(d)[3:]
            for fp in glob.glob(d + "/*.dcm"):
                h = pydicom.dcmread(fp, stop_before_pixels=True)
                normal = np.cross(*np.array(h.ImageOrientationPatient, float).reshape(2, 3)); pos = np.array(h.ImagePositionPatient, float)
                insts.append({"series": sname, "acq": str(h.get("AcquisitionNumber", "")), "sop": str(h.SOPInstanceUID), "pos": pos, "z": float(np.dot(pos, normal)), "file": fp, "rows": h.Rows, "cols": h.Columns})
                if str(h.SOPInstanceUID) in ref:
                    per_series[sname] += 1
        if not per_series:
            row["status"], row["detail"] = "UNRESOLVED", "none of the SEG's referenced instances exists in the study"; log.append(row); continue
        best = per_series.most_common(1)[0][0]; row["chosen_series"] = best[-8:]
        sub = [i for i in insts if i["series"] == best]
        if (sub[0]["rows"], sub[0]["cols"]) != frames.shape[1:]:
            row["status"], row["detail"] = "UNRESOLVED", "SEG frame size differs from the CT"; log.append(row); continue
        cand = []
        for k in sorted({i["acq"] for i in sub}):
            P = np.array([i["pos"] for i in sub if i["acq"] == k])
            frac = float(np.mean([np.linalg.norm(P - p, axis=1).min() < 0.5 for p in fpos]))
            if frac >= 0.95:
                cand.append(k)
            row["detail"] += f"acq{k}: {frac:.2f} of SEG frames at an instance position; "
        row["candidates"] = ",".join(cand)
        info = {"ct_series": best, "seg_series": meta["seg_series"], "old_converter_series": meta["ct_series"]}
        if len(cand) == 0:
            row["status"] = "UNRESOLVED"; row["detail"] += "| no acquisition matches the SEG frame positions"
        elif len(cand) == 1:
            row["status"] = "RESOLVED"; write_volume(os.path.join(out, case), [i for i in sub if i["acq"] == cand[0]], frames, fpos, fseg, segs, {**info, "acquisition": cand[0]})
        else:
            row["status"] = "AMBIGUOUS"
            for k in cand:
                write_volume(os.path.join(out, f"{case}_acq{k}"), [i for i in sub if i["acq"] == k], frames, fpos, fseg, segs, {**info, "acquisition": k})
    except Exception as e:
        row["status"], row["detail"] = "UNRESOLVED", f"{type(e).__name__}: {str(e)[:100]}"
    log.append(row); print(row["case"][:12], row["status"], "| cand:", row["candidates"], "| used by old converter:", row["used_by_old_converter"], "| chosen:", row["chosen_series"], flush=True)
os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
print(dict(collections.Counter(r["status"] for r in log)))
