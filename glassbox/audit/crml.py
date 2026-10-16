"""CRML-CT (TCIA Colorectal-Liver-Metastases) loader: CT series to HU, DICOM-SEG to masks. Segment roles come from SegmentLabel and the series from
SegmentAlgorithmType / SeriesDescription, never from segment order (results/w1_training_audit/DEVIATIONS.md #1)."""
import glob
import os
import re

import numpy as np
import pydicom

MANUAL = "MANUAL"


def roles(seg_ds):
    """{segment_number: 'liver' | 'liver_remnant' | 'portal_vein' | 'hepatic_vein' | 'tumour' | 'other'} from the segment label (exact, case-insensitive). Note 'Liver' and 'Liver Remnant' share the type code meaning 'Liver'."""
    out = {}
    for s in seg_ds.SegmentSequence:
        lab = str(s.SegmentLabel).strip().lower()
        out[int(s.SegmentNumber)] = ("liver" if lab == "liver" else "liver_remnant" if lab == "liver remnant" else "portal_vein" if lab == "portal" else "hepatic_vein" if lab == "hepatic" else "tumour" if re.fullmatch(r"tumou?r_\d+", lab) else "other")
    return out


def choose_seg_series(patient_dir):
    """The expert series: every segment MANUAL. Returns (path, info) or (None, reason). Automatic and semi-automatic (AI or AI-corrected) series are never used."""
    cands, seen = [], []
    for sdir in sorted(glob.glob(os.path.join(patient_dir, "SEG_*"))):
        fs = glob.glob(os.path.join(sdir, "*.dcm"))
        if len(fs) != 1:
            seen.append(f"{os.path.basename(sdir)}: {len(fs)} files"); continue
        d = pydicom.dcmread(fs[0], stop_before_pixels=True)
        algs = {str(s.get("SegmentAlgorithmType", "")) for s in d.SegmentSequence}
        seen.append(f"{str(d.get('SeriesDescription', ''))}[{'/'.join(sorted(algs))}]")
        if algs == {MANUAL}:
            cands.append(fs[0])
    if len(cands) != 1:
        return None, f"{len(cands)} manual-only series among: {'; '.join(seen)}"
    return cands[0], "; ".join(seen)


def load_ct(patient_dir):
    """CT volume in HU, int16, axes [row, col, slice] sorted along the slice normal; spacing (row, col, slice) in mm; SOP UID per slice; max deviation of slice gaps."""
    dirs = sorted(glob.glob(os.path.join(patient_dir, "CT_*")))
    if len(dirs) != 1:
        raise ValueError(f"{len(dirs)} CT series")
    ds = [pydicom.dcmread(f) for f in glob.glob(os.path.join(dirs[0], "*.dcm"))]
    ds = [d for d in ds if "PixelData" in d]
    iop = np.array(ds[0].ImageOrientationPatient, float); n = np.cross(iop[:3], iop[3:])
    pos = np.array([float(np.dot(np.array(d.ImagePositionPatient, float), n)) for d in ds]); order = np.argsort(pos); ds = [ds[i] for i in order]; pos = pos[order]
    vol = np.stack([(d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1)) + float(d.get("RescaleIntercept", 0))) for d in ds], axis=-1)
    gaps = np.diff(pos); dz = float(np.median(gaps)); ps = [float(x) for x in ds[0].PixelSpacing]
    return np.clip(vol, -32768, 32767).astype(np.int16), (ps[0], ps[1], abs(dz)), [str(d.SOPInstanceUID) for d in ds], float(np.abs(gaps - dz).max()), pos


def load_seg_masks(seg_path, sop_uids, shape):
    """{segment_number: bool volume [row, col, slice]} aligned to the CT; frames are placed by their referenced CT SOP instance; an unplaceable frame raises (the patient is excluded and logged)."""
    d = pydicom.dcmread(seg_path)
    arr = d.pixel_array.astype(bool)
    index = {u: i for i, u in enumerate(sop_uids)}
    masks = {int(s.SegmentNumber): np.zeros(shape, bool) for s in d.SegmentSequence}
    for k, fg in enumerate(d.PerFrameFunctionalGroupsSequence):
        seg_no = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        z = None
        try:
            z = index.get(str(fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID))
        except Exception:
            pass
        if z is None:
            raise ValueError("frame cannot be placed on a CT slice")
        masks[seg_no][:, :, z] |= arr[k]
    return masks
