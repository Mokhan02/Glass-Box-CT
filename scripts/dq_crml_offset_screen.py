"""W1 CRML-CT header screen: SEG frame positions vs the CT instances they reference, for ALL patients (not only the included ones). `glassbox.audit.crml.load_seg_masks` places each SEG frame on the CT slice by its referenced SOP instance
UID and never reads the SEG's own ImagePositionPatient, i.e. the same effective behaviour as `ignore_spatial_locations=True`. This script chooses the same manual SEG series (choose_seg_series), reads the headers only, and reports per patient the in-plane (array x / y) and through-plane
offset between each frame's ImagePositionPatient and the referenced CT instance's, whether the SEG's referenced series equals the CT series, and the FrameOfReferenceUID match. Flag: in-plane offset > 1 mm.
Usage: python scripts/dq_crml_offset_screen.py <crml root> <crml_inventory.csv> <out csv>"""
import csv
import glob
import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, ".")
from glassbox.audit.crml import choose_seg_series  # noqa: E402
from glassbox.data.seg_offset import frame_offsets_mm  # noqa: E402

root, inv, out = sys.argv[1:4]; inc = {r["patient"]: r["included"] for r in csv.DictReader(open(inv))}; rows = []
for p in sorted(os.listdir(root)):
    pd_ = os.path.join(root, p); r = {"patient": p, "in_w1_inventory_included": inc.get(p, "")}
    seg_path, info = choose_seg_series(pd_)
    if seg_path is None:
        rows.append({**r, "status": "no single manual SEG series", "detail": info[:120]}); continue
    cts = sorted(glob.glob(os.path.join(pd_, "CT_*")))
    if len(cts) != 1:
        rows.append({**r, "status": f"{len(cts)} CT series"}); continue
    ct = {str(h.SOPInstanceUID): h for h in (pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(cts[0] + "/*.dcm"))}; ct0 = next(iter(ct.values())); seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    f_ipp, c_ipp = [], []
    for f in seg.PerFrameFunctionalGroupsSequence:
        try:
            s = str(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        except Exception:
            continue
        if s in ct:
            f_ipp.append(f.PlanePositionSequence[0].ImagePositionPatient); c_ipp.append(ct[s].ImagePositionPatient)
    if not f_ipp:
        rows.append({**r, "status": "no frame references a CT instance"}); continue
    dx, dy, dn = frame_offsets_mm(np.array(f_ipp, float), np.array(c_ipp, float), ct0.ImageOrientationPatient)
    rows.append({**r, "status": "ok", "n_frames": len(seg.PerFrameFunctionalGroupsSequence), "n_matched": len(f_ipp), "x_median_mm": round(float(np.median(dx)), 3), "x_max_abs_mm": round(float(np.abs(dx).max()), 3), "y_median_mm": round(float(np.median(dy)), 3),
                 "y_max_abs_mm": round(float(np.abs(dy).max()), 3), "through_max_abs_mm": round(float(np.abs(dn).max()), 3), "frame_of_reference_equal": str(seg.FrameOfReferenceUID) == str(ct0.FrameOfReferenceUID),
                 "flag_inplane_gt_1mm": bool(max(np.abs(dx).max(), np.abs(dy).max()) > 1.0)})
    print(p, rows[-1]["x_max_abs_mm"], rows[-1]["y_max_abs_mm"], rows[-1]["through_max_abs_mm"], flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r["status"] == "ok"]; print(f"{len(rows)} patients, {len(ok)} screened, flagged {sum(r['flag_inplane_gt_1mm'] for r in ok)}; not screened: {[(r['patient'], r['status']) for r in rows if r['status'] != 'ok']}")
