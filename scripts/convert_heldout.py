"""Convert held-out DICOM (IDC layout: <collection>/<patient>/<study>/<Modality>_<SeriesUID>/*.dcm) to NIfTI.
For every ground-truth DICOM-SEG: find the CT series it references, write ct.nii.gz and seg.nii.gz (segment k = label k)
plus segments.json. Only the SEG whose SeriesDescription equals --seg-description is used: HCC-TACE-Seg also ships
AI-generated segmentations ('AIMI liver-tumor AI segmentation') that must NOT be treated as ground truth.
Usage: python scripts/convert_heldout.py --collection c4kc_kits --out cache/heldout_nii [--limit 5] [--patients KiTS-00000 ...]"""
import argparse
import glob
import json
import logging
import os
import sys

import highdicom as hd
import numpy as np
import pydicom
import SimpleITK as sitk

logging.getLogger("highdicom").setLevel(logging.CRITICAL)  # it logs one line per unmatched source SOP


def sorted_ct(ct_dir):
    files = sorted(glob.glob(ct_dir + "/*.dcm"))
    meta = [(pydicom.dcmread(f, stop_before_pixels=True), f) for f in files]
    normal = np.cross(*np.array(meta[0][0].ImageOrientationPatient, float).reshape(2, 3))
    meta.sort(key=lambda m: float(np.dot(np.array(m[0].ImagePositionPatient, float), normal)))
    return meta, normal


def label_by_geometry(seg_file, ct_dirs, min_matched=0.95):
    """Fallback when the SEG's source-instance UIDs do not resolve (HCC-TACE-Seg). Place each SEG frame on the CT slice at
    the same physical position. Picks the CT series that matches the most frames; fails unless >= min_matched match."""
    ds = pydicom.dcmread(seg_file)
    frames = ds.pixel_array  # [frames, rows, cols]
    if frames.ndim == 2:
        frames = frames[None]
    fpos = np.array([np.array(f.PlanePositionSequence[0].ImagePositionPatient, float) for f in ds.PerFrameFunctionalGroupsSequence])
    fseg = np.array([f.SegmentIdentificationSequence[0].ReferencedSegmentNumber for f in ds.PerFrameFunctionalGroupsSequence])
    best = None
    for d in ct_dirs:
        meta, normal = sorted_ct(d)
        if (meta[0][0].Rows, meta[0][0].Columns) != frames.shape[1:]:
            continue
        pos = np.array([np.array(m.ImagePositionPatient, float) for m, _ in meta])
        tol = 0.6 * float(np.median(np.abs(np.diff(pos @ normal)))) if len(pos) > 1 else 0.6
        idx = np.full(len(fpos), -1)
        for k, p in enumerate(fpos):
            dist = np.linalg.norm(pos - p, axis=1)
            j = int(dist.argmin())
            if dist[j] <= max(tol, 0.5):
                idx[k] = j
        frac = float((idx >= 0).mean())
        if best is None or frac > best[0]:
            best = (frac, d, meta, idx)
    if best is None or best[0] < min_matched:
        raise RuntimeError(f"geometry match failed: best matched fraction {0.0 if best is None else best[0]:.2f}")
    frac, d, meta, idx = best
    label = np.zeros((len(meta),) + frames.shape[1:], np.uint8)
    for k in np.flatnonzero(idx >= 0):
        label[idx[k]][frames[k] > 0] = fseg[k]
    return label, d, meta, frac


ap = argparse.ArgumentParser()
ap.add_argument("--root", default="cache/heldout")
ap.add_argument("--collection", required=True)
ap.add_argument("--out", default="cache/heldout_nii")
ap.add_argument("--seg-description", default="Segmentation")
ap.add_argument("--patients", nargs="*")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
base = os.path.join(a.root, a.collection)
patients = a.patients or sorted(os.listdir(base))
done = 0
for pid in patients:
    for study in sorted(glob.glob(f"{base}/{pid}/*")):
        for seg_dir in sorted(glob.glob(f"{study}/SEG_*")):
            seg_file = sorted(glob.glob(seg_dir + "/*.dcm"))[0]
            hdr = pydicom.dcmread(seg_file, stop_before_pixels=True)
            if hdr.get("SeriesDescription") != a.seg_description:
                continue
            case = f"{pid}_{os.path.basename(study)[-6:]}"
            dst = os.path.join(a.out, a.collection, case)
            if os.path.exists(os.path.join(dst, ".done")):
                continue
            try:
                segs = [(sg.SegmentNumber, sg.SegmentLabel) for sg in pydicom.dcmread(seg_file).SegmentSequence]
                method, frac = "uid", 1.0
                try:
                    seg = hd.seg.segread(seg_file)
                    ref = {s for _, s, _ in seg.get_source_image_uids()}
                    ct_dirs = [d for d in glob.glob(f"{study}/CT_*") if os.path.basename(d)[3:] in ref]
                    if len(ct_dirs) != 1:
                        raise RuntimeError("SEG references %d CT series" % len(ct_dirs))
                    meta, _ = sorted_ct(ct_dirs[0])
                    sop = [m.SOPInstanceUID for m, _ in meta]
                    label = np.zeros((len(meta), meta[0][0].Rows, meta[0][0].Columns), np.uint8)
                    for num, name in segs:
                        px = seg.get_pixels_by_source_instance(sop, segment_numbers=[num], combine_segments=False,
                                                               assert_missing_frames_are_empty=False,
                                                               ignore_spatial_locations=True)  # SEG matched to its CT by UID
                        label[px[..., 0] > 0] = num
                    ct_dir = ct_dirs[0]
                except Exception:
                    method = "geometry"
                    label, ct_dir, meta, frac = label_by_geometry(seg_file, glob.glob(f"{study}/CT_*"))
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames([f for _, f in meta])
                ct = reader.Execute()
                lab = sitk.GetImageFromArray(label)
                lab.CopyInformation(ct)
                os.makedirs(dst, exist_ok=True)
                sitk.WriteImage(ct, os.path.join(dst, "ct.nii.gz"))
                sitk.WriteImage(lab, os.path.join(dst, "seg.nii.gz"))
                json.dump({"segments": {str(n): l for n, l in segs}, "ct_series": os.path.basename(ct_dir), "label_method": method, "frames_matched": frac,
                           "seg_series": os.path.basename(seg_dir), "n_slices": len(meta)}, open(os.path.join(dst, "segments.json"), "w"))
                open(os.path.join(dst, ".done"), "w").close()
                print(f"{case}: {len(meta)} slices ({method}, matched {frac:.2f}), segments {segs}, voxels per segment "
                      + str({n: int((label == n).sum()) for n, _ in segs}), flush=True)
            except Exception as e:  # keep going; the failure is part of the record
                print(f"FAILED {case}: {type(e).__name__}: {str(e)[:150]}", flush=True)
            done += 1
    if a.limit and done >= a.limit:
        break
