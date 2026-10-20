"""DICOM-SEG placement that honours the SEG's declared spatial locations (fix of the HCC_019_773068 displacement, see results/dq_hcc_labels/DQ_REPORT.md).
The UID-route converter used to read SEG pixels with `ignore_spatial_locations=True`, which silently discards a constant offset between the SEG frames' ImagePositionPatient and the CT instances they reference. `read_uid_label` reads by source-instance UID and ALWAYS applies the declared in-plane offset explicitly (rounded to whole voxels), refusing if the offset is not constant across frames; highdicom's own spatial-location
check was found not to compare in-plane positions, so dropping the flag alone would not have changed the HCC_019 result.
Pure helpers (`frame_offsets_mm`, `offset_voxels`, `shift_label`) need only numpy; highdicom / pydicom are imported lazily."""
from __future__ import annotations

import numpy as np

OFFSET_TOL_MM = 0.5          # offsets that differ across frames by more than this are not a rigid shift


def frame_offsets_mm(frame_ipp, ct_ipp, iop):
    """Per frame: (SEG ImagePositionPatient - referenced CT ImagePositionPatient) projected on the CT's column-index direction (array x), row-index direction (array y) and the slice normal. Inputs: [n, 3] arrays and the 6-vector IOP."""
    iop = np.asarray(iop, float); d = np.asarray(frame_ipp, float) - np.asarray(ct_ipp, float); normal = np.cross(iop[:3], iop[3:])
    return d @ iop[:3], d @ iop[3:], d @ normal


def offset_voxels(dx_mm, dy_mm, pixel_spacing, tol_mm=OFFSET_TOL_MM):
    """Whole-voxel in-plane shift (sy along rows, sx along columns) from per-frame offsets; ValueError if the offset is not constant across frames. pixel_spacing = DICOM (row spacing, column spacing)."""
    dx, dy = np.asarray(dx_mm, float), np.asarray(dy_mm, float)
    if max(np.ptp(dx), np.ptp(dy)) > tol_mm:
        raise ValueError("SEG offset is not constant across frames")
    return int(round(float(np.median(dy)) / float(pixel_spacing[0]))), int(round(float(np.median(dx)) / float(pixel_spacing[1])))


def shift_label(label, sy, sx):
    """label [slices, rows, cols]: a SEG pixel at (r, c) moves to CT pixel (r + sy, c + sx); zero fill."""
    out = np.zeros_like(label); R, C = label.shape[1:]
    out[:, max(sy, 0):R + min(sy, 0), max(sx, 0):C + min(sx, 0)] = label[:, max(-sy, 0):R + min(-sy, 0), max(-sx, 0):C + min(-sx, 0)]
    return out


def read_uid_label(seg_file, ct_headers):
    """Label volume [slices, rows, cols] (CT order = ImagePositionPatient along the normal) for a SEG that references the CT series by SOP UID. ct_headers: list of (pydicom header, path) sorted along the normal.
    The pixels are read by source-instance UID (strictly when highdicom accepts that, otherwise ignoring its spatial-location check), and the SEG's declared in-plane offset from the referenced CT instances is ALWAYS computed from the headers and applied (whole voxels),
    because highdicom's own check does not compare in-plane positions (shown by tests/test_seg_offset.py: a SEG 5 mm off its CT is accepted as is). Refuses if the offset is not constant across frames or no frame references the CT.
    Returns (label, info), info = {"mode": "matched" | "declared_offset", "shift_voxels": (sy, sx), "offset_mm": (dx, dy)}; "declared_offset" means a non-zero whole-voxel shift was applied."""
    import highdicom as hd
    import pydicom
    ds = pydicom.dcmread(seg_file); segs = [(s.SegmentNumber, s.SegmentLabel) for s in ds.SegmentSequence]; seg = hd.seg.segread(seg_file); sop = [m.SOPInstanceUID for m, _ in ct_headers]
    label = np.zeros((len(ct_headers), ct_headers[0][0].Rows, ct_headers[0][0].Columns), np.uint8)

    def fill(ignore):
        label[:] = 0
        for num, _ in segs:
            px = seg.get_pixels_by_source_instance(sop, segment_numbers=[num], combine_segments=False, assert_missing_frames_are_empty=False, ignore_spatial_locations=ignore); label[px[..., 0] > 0] = num
    try:
        fill(False)
    except Exception as e:      # highdicom refuses on some spatial mismatches (not on in-plane offsets); any other error is real
        if "spatial" not in str(e).lower() and "position" not in str(e).lower():
            raise
        fill(True)
    ipp = {str(m.SOPInstanceUID): np.array(m.ImagePositionPatient, float) for m, _ in ct_headers}; iop = np.array(ct_headers[0][0].ImageOrientationPatient, float); f_ipp, c_ipp = [], []
    for f in ds.PerFrameFunctionalGroupsSequence:
        for s in (s for d in f.get("DerivationImageSequence", []) for s in d.get("SourceImageSequence", [])):
            if str(s.ReferencedSOPInstanceUID) in ipp:
                f_ipp.append(np.array(f.PlanePositionSequence[0].ImagePositionPatient, float)); c_ipp.append(ipp[str(s.ReferencedSOPInstanceUID)]); break
    if not f_ipp:
        raise ValueError("no SEG frame references a CT instance of this series")
    dx, dy, _ = frame_offsets_mm(f_ipp, c_ipp, iop); sy, sx = offset_voxels(dx, dy, [float(x) for x in ct_headers[0][0].PixelSpacing]); info = {"mode": "declared_offset" if (sy, sx) != (0, 0) else "matched", "shift_voxels": (sy, sx), "offset_mm": (float(np.median(dx)), float(np.median(dy)))}
    return (shift_label(label, sy, sx) if (sy, sx) != (0, 0) else label), info
