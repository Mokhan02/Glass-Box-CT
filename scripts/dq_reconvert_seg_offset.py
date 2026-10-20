"""Corrected re-conversion of a UID-route HCC case that honours the DICOM-SEG's own spatial metadata (CPU only).
Cause found by scripts/dq_source_check.py: in HCC_019_773068 the manual SEG references the CT series we used (same SeriesInstanceUID and FrameOfReferenceUID) but every SEG frame's ImagePositionPatient is offset by a constant 25.0 mm from the CT instance it references.
scripts/convert_heldout.py reads the SEG pixels with `get_pixels_by_source_instance(..., ignore_spatial_locations=True)`, i.e. it discards that offset and places each frame by index. This script repeats the original conversion and then shifts the label in-plane by the SEG-declared
offset (rounded to whole voxels; the rounding error is at most half a voxel, 0.35 mm at 0.703 mm), refusing to proceed if the offset is not constant across frames. The CT is not changed (symlink); the original conversion stays untouched for the sensitivity analysis.
Usage (cwd = cache root): python scripts/dq_reconvert_seg_offset.py <case dir name> <out root>"""
import glob
import json
import os
import re
import sys

import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk

case, out_root = sys.argv[1:3]
src = f"heldout_nii/hcc_tace_seg/{case}"; meta = json.load(open(f"{src}/segments.json")); assert meta["label_method"] == "uid"
pid = re.match(r"^(HCC_\d+)", case).group(1); seg_dir = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/{meta['seg_series']}")[0]; ct_dir = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/{meta['ct_series']}")[0]
seg_file = sorted(glob.glob(seg_dir + "/*.dcm"))[0]


def sorted_ct(d):
    ms = [(pydicom.dcmread(f, stop_before_pixels=True), f) for f in sorted(glob.glob(d + "/*.dcm"))]; n = np.cross(*np.array(ms[0][0].ImageOrientationPatient, float).reshape(2, 3))
    ms.sort(key=lambda m: float(np.dot(np.array(m[0].ImagePositionPatient, float), n))); return ms


ds = pydicom.dcmread(seg_file); segs = [(sg.SegmentNumber, sg.SegmentLabel) for sg in ds.SegmentSequence]; seg = hd.seg.segread(seg_file); ms = sorted_ct(ct_dir); sop = [m.SOPInstanceUID for m, _ in ms]
label = np.zeros((len(ms), ms[0][0].Rows, ms[0][0].Columns), np.uint8)
for num, name in segs:      # the original conversion, unchanged
    px = seg.get_pixels_by_source_instance(sop, segment_numbers=[num], combine_segments=False, assert_missing_frames_are_empty=False, ignore_spatial_locations=True); label[px[..., 0] > 0] = num
assert int((label == 2).sum()) == int((np.asarray(nib.load(f"{src}/seg.nii.gz").dataobj) == 2).sum()), "re-conversion does not reproduce the stored original label"
iop = np.array(ms[0][0].ImageOrientationPatient, float); col_dir_x, row_dir_y = iop[:3], iop[3:]; ct_ipp = {str(m.SOPInstanceUID): np.array(m.ImagePositionPatient, float) for m, _ in ms}
dx, dy = [], []
for f in ds.PerFrameFunctionalGroupsSequence:
    ipp = np.array(f.PlanePositionSequence[0].ImagePositionPatient, float)
    for s in (s for d in f.get("DerivationImageSequence", []) for s in d.get("SourceImageSequence", [])):
        if str(s.ReferencedSOPInstanceUID) in ct_ipp:
            v = ipp - ct_ipp[str(s.ReferencedSOPInstanceUID)]; dx.append(float(v @ col_dir_x)); dy.append(float(v @ row_dir_y)); break
assert max(np.ptp(dx), np.ptp(dy)) < 0.5, "SEG offset is not constant across frames"
ps_row, ps_col = (float(x) for x in ms[0][0].PixelSpacing); sy, sx = int(round(np.median(dy) / ps_row)), int(round(np.median(dx) / ps_col))   # CT row index of a SEG pixel (r, c) = r + sy, c + sx
shifted = np.zeros_like(label); R, C = label.shape[1:]
shifted[:, max(sy, 0):R + min(sy, 0), max(sx, 0):C + min(sx, 0)] = label[:, max(-sy, 0):R + min(-sy, 0), max(-sx, 0):C + min(-sx, 0)]
lab = sitk.GetImageFromArray(shifted); lab.CopyInformation(sitk.ReadImage(f"{src}/ct.nii.gz")); dst = f"{out_root}/hcc_tace_seg/{case}"; os.makedirs(dst, exist_ok=True)
sitk.WriteImage(lab, f"{dst}/seg.nii.gz")
if not os.path.lexists(f"{dst}/ct.nii.gz"):
    os.symlink(os.path.abspath(f"{src}/ct.nii.gz"), f"{dst}/ct.nii.gz")
json.dump({**meta, "label_method": "uid+seg_declared_offset", "seg_declared_offset_mm": {"along_column_index_x": round(float(np.median(dx)), 3), "along_row_index_y": round(float(np.median(dy)), 3)}, "applied_shift_voxels": {"x": sx, "y": sy},
           "voxels_per_segment_original": {str(n): int((label == n).sum()) for n, _ in segs}, "voxels_per_segment_corrected": {str(n): int((shifted == n).sum()) for n, _ in segs}}, open(f"{dst}/segments.json", "w"), indent=1)
print(case, "offset mm", np.median(dx), np.median(dy), "shift voxels x,y", sx, sy, "voxels kept", {n: (int((label == n).sum()), int((shifted == n).sum())) for n, _ in segs}, flush=True)
