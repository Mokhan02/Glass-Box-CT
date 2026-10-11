"""What are the duplicate slices in the 13 series with repeated positions? For each such case compare, within each duplicated position, the header
fields that distinguish reconstructions/phases and the pixel data. Usage: python scripts/hcc_duplicate_inspect.py cache/hcc_phase_check.csv"""
import collections
import csv
import glob
import json
import re
import sys

import numpy as np
import pydicom

root, held = "cache/heldout_nii/hcc_tace_seg", "cache/heldout/hcc_tace_seg"
FIELDS = ["InstanceNumber", "SliceThickness", "ConvolutionKernel", "AcquisitionNumber", "ImageType", "SeriesNumber", "ContentTime", "TriggerTime", "SpacingBetweenSlices", "XRayTubeCurrent", "KVP"]
for r in [r for r in csv.DictReader(open(sys.argv[1])) if int(r["n_duplicate_slices"]) > 0][:13]:
    case = r["case"]; pid = re.match(r"^(HCC_\d+)", case).group(1); meta = json.load(open(f"{root}/{case}/segments.json"))
    d = glob.glob(f"{held}/{pid}/*/{meta['ct_series']}")[0]
    ds = [pydicom.dcmread(f) for f in sorted(glob.glob(d + "/*.dcm"))]
    n = np.cross(*np.array(ds[0].ImageOrientationPatient, float).reshape(2, 3))
    groups = collections.defaultdict(list)
    for h in ds:
        groups[round(float(np.dot(np.array(h.ImagePositionPatient, float), n)), 1)].append(h)
    dup = [g for g in groups.values() if len(g) > 1]
    ident = diffs = 0; varying = collections.Counter(); sizes = collections.Counter(len(g) for g in groups.values())
    for g in dup:
        a, b = g[0], g[1]
        same = np.array_equal(a.pixel_array, b.pixel_array); ident += same; diffs += (not same)
        for k in FIELDS:
            if str(a.get(k, "")) != str(b.get(k, "")):
                varying[k] += 1
    absdiff = np.mean([np.abs(g[0].pixel_array.astype(float) - g[1].pixel_array.astype(float)).mean() for g in dup[:10]])
    print(f"{case[:11]} {r['route']:8s} files {len(ds)} unique z {len(groups)} copies-per-position {dict(sizes)} | identical pixel pairs {ident}/{len(dup)} | mean |diff| {absdiff:.1f} HU | headers that differ between copies: {dict(varying)}")
