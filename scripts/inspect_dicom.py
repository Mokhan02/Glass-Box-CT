"""List the series of one held-out patient: modality, slice count, and for SEG the segment labels.
Usage: python scripts/inspect_dicom.py cache/heldout/c4kc_kits/KiTS-00000"""
import glob
import sys

import pydicom

patient = sys.argv[1]
for d in sorted(glob.glob(f"{patient}/*/*")):
    files = sorted(glob.glob(d + "/*.dcm"))
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    print(f"{d.split('/')[-1][:40]:42s} {ds.Modality:4s} files={len(files):4d} desc={ds.get('SeriesDescription', '-')!r}")
    if ds.Modality == "SEG":
        full = pydicom.dcmread(files[0])
        print("   segments:", [(s.SegmentNumber, s.SegmentLabel) for s in full.SegmentSequence], "frames:", full.NumberOfFrames)
