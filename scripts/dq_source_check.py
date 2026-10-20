"""Data-quality source check (CPU only): is a label displacement in the source DICOM-SEG or introduced by our conversion?
For each HCC case: read the manual SEG headers (the series recorded in the case's segments.json) and the CT series our converter used; compare (1) the SEG's referenced SeriesInstanceUID with the CT series used, (2) FrameOfReferenceUID of SEG and CT,
(3) per-frame ImagePositionPatient of every SEG frame with the ImagePositionPatient of the CT instance it references by SOPInstanceUID (in-plane components along the CT row / column directions, and through-plane), (4) Rows, Columns, PixelSpacing and
ImageOrientationPatient, (5) the label method and whether the case is among the geometry-route cases touched by the converter fixes (results/exp1_audit/hcc_clean_conversion.csv).
Our UID-route converter reads the SEG pixels by referenced SOP UID with `ignore_spatial_locations=True`, i.e. it does NOT use the SEG's own positions: a displacement that exists only in the SEG's positions would be missed by that converter.
Usage: python scripts/dq_source_check.py <out json> <case dir name> [<case dir name> ...]   (cwd = cache root containing heldout/ heldout_nii/ heldout_nii_clean/)"""
import glob
import json
import os
import re
import sys

import numpy as np
import pydicom

out, cases = sys.argv[1], sys.argv[2:]
R = {}
for case in cases:
    root = "heldout_nii_clean" if os.path.isdir(f"heldout_nii_clean/hcc_tace_seg/{case}") else "heldout_nii"
    meta = json.load(open(f"{root}/hcc_tace_seg/{case}/segments.json")); pid = re.match(r"^(HCC_\d+)", case).group(1)
    seg_series = meta["seg_series"].replace("SEG_", ""); ct_series = meta["ct_series"].replace("CT_", "")
    sd = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/SEG_{seg_series}"); cd = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/CT_{ct_series}")
    rec = {"nii_root": root, "label_method": meta.get("label_method", "uid"), "frames_matched": meta.get("frames_matched")}
    if not sd or not cd:
        R[case] = {**rec, "error": "series directory not found"}; continue
    seg = pydicom.dcmread(sorted(glob.glob(sd[0] + "/*.dcm"))[0], stop_before_pixels=True)
    ct = {str(h.SOPInstanceUID): h for h in (pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(cd[0] + "/*.dcm"))}; ct0 = next(iter(ct.values()))
    ref_series = {str(s.SeriesInstanceUID) for s in seg.get("ReferencedSeriesSequence", [])}
    iop = np.array(ct0.ImageOrientationPatient, float); row_dir, col_dir = iop[:3], iop[3:]; normal = np.cross(row_dir, col_dir)      # row_dir = direction of increasing COLUMN index (array x); col_dir = direction of increasing ROW index (array y)
    shared = seg.SharedFunctionalGroupsSequence[0]; s_iop = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient, float) if "PlaneOrientationSequence" in shared else None
    s_ps = [float(x) for x in shared.PixelMeasuresSequence[0].PixelSpacing] if "PixelMeasuresSequence" in shared else None
    d_row, d_col, d_n, matched, n = [], [], [], 0, 0
    for f in seg.PerFrameFunctionalGroupsSequence:
        n += 1; ipp = np.array(f.PlanePositionSequence[0].ImagePositionPatient, float); sops = [str(s.ReferencedSOPInstanceUID) for d in f.get("DerivationImageSequence", []) for s in d.get("SourceImageSequence", [])]
        hit = [ct[s] for s in sops if s in ct]
        if hit:
            matched += 1; dv = ipp - np.array(hit[0].ImagePositionPatient, float); d_row.append(float(dv @ row_dir)); d_col.append(float(dv @ col_dir)); d_n.append(float(dv @ normal))
    f_ = lambda v: None if not v else {"median": round(float(np.median(v)), 2), "max_abs": round(float(np.max(np.abs(v))), 2)}      # noqa: E731
    rec.update({"series_ref_equals_ct_used": ct_series in ref_series, "seg_referenced_series": sorted(ref_series), "ct_series_used": ct_series, "frame_of_reference_equal": str(seg.FrameOfReferenceUID) == str(ct0.FrameOfReferenceUID),
                "n_frames": n, "n_frames_with_referenced_ct_instance": matched, "delta_ipp_along_column_index_dir_x_mm": f_(d_row), "delta_ipp_along_row_index_dir_y_mm": f_(d_col), "delta_ipp_through_plane_mm": f_(d_n),
                "seg_rows_cols": [int(seg.Rows), int(seg.Columns)], "ct_rows_cols": [int(ct0.Rows), int(ct0.Columns)], "seg_pixel_spacing": s_ps, "ct_pixel_spacing": [float(x) for x in ct0.PixelSpacing],
                "iop_max_abs_diff": None if s_iop is None else round(float(np.abs(s_iop - iop).max()), 6), "seg_description": str(seg.get("SeriesDescription", ""))})
    rec["flag_inplane_shift_gt_1mm"] = bool(matched and max(abs(rec["delta_ipp_along_column_index_dir_x_mm"]["max_abs"]), abs(rec["delta_ipp_along_row_index_dir_y_mm"]["max_abs"])) > 1.0)
    R[case] = rec; print(case, rec["label_method"], "ref_ok", rec["series_ref_equals_ct_used"], "FoR", rec["frame_of_reference_equal"], "dIPP x/y/normal", rec["delta_ipp_along_column_index_dir_x_mm"], rec["delta_ipp_along_row_index_dir_y_mm"], rec["delta_ipp_through_plane_mm"], flush=True)
json.dump(R, open(out, "w"), indent=1)
