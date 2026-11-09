"""W4b: TotalSegmentator `liver_vessels` task (class `liver_tumor`) on every series of the listed WAW-TACE patients, in the W4 prediction format (results/w4b_waw/PREREG.md section 3; held-out status inferred, not author-confirmed: DEVIATIONS 25).
Per series: (1) task `liver_vessels` (multilabel, the tumour = the label named `liver_tumor` in the installed version's class map), (2) task `total` with `roi_subset=["liver"]` for the liver of the alignment gate (this task has no liver class). Both are returned on the input CT grid; the shape is asserted equal to the CT's.
Writes <out>/<case>.npz with bool arrays `tumour` and `liver` (skipped when present: resumable) and <out>/ts_meta.json (versions, class maps, weights file names). Python of the TotalSegmentator environment (`venv_ts`), cwd anywhere.
Usage: python scripts/w6_ts_run.py <gt root> <patients file> <out dir> [--limit N] [--device gpu|cpu] [--cases 14_1 540_1]"""
import json
import os
import sys
import time

import numpy as np


def mask_from_multilabel(arr, class_map, name):
    """class_map: {int label: name} of the task; returns the bool mask of the label called `name` (raises if absent: a version without the class must not silently give an empty mask)."""
    ids = [int(k) for k, v in class_map.items() if v == name]
    if not ids:
        raise KeyError(f"class {name!r} not in the task's class map {sorted(class_map.values())}")
    return np.isin(np.asarray(arr), ids)


def main(argv):
    gt, pfile, out = argv[:3]; opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    dev = opt("--device", "gpu"); limit = int(opt("--limit", 0)) or None; os.makedirs(out, exist_ok=True)
    import nibabel as nib
    import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator as ts
    pats = set(open(pfile).read().split()); cases = sorted((d for d in os.listdir(gt) if d[0].isdigit() and d.split("_")[0] in pats), key=lambda d: (int(d.split("_")[0]), d)); sel = opt("--cases")
    if sel:
        cases = [c for c in argv[argv.index("--cases") + 1:] if c in cases] if "--cases" in argv else cases
    cases = cases[:limit] if limit else cases
    lv, tot = class_map["liver_vessels"], class_map["total"]; t0 = time.time(); n = 0
    json.dump({"totalsegmentator_version": getattr(totalsegmentator, "__version__", None), "class_map_liver_vessels": {str(k): v for k, v in lv.items()}, "liver_label_in_total": [k for k, v in tot.items() if v == "liver"], "device": dev,
               "held_out_status": "inferred, not author-confirmed (Dataset008_HepaticVessel = MSD Task 08; DEVIATIONS 25)"}, open(os.path.join(out, "ts_meta.json"), "w"), indent=1)
    for i, c in enumerate(cases, 1):
        fn = os.path.join(out, c + ".npz")
        if os.path.exists(fn):
            continue
        path = os.path.join(gt, c, "ct.nii.gz"); shape = tuple(nib.load(path).shape)
        v = np.asarray(ts(path, None, ml=True, task="liver_vessels", device=dev, quiet=True).dataobj); L = np.asarray(ts(path, None, ml=True, task="total", roi_subset=["liver"], device=dev, quiet=True).dataobj)
        assert v.shape == shape and L.shape == shape, (c, v.shape, L.shape, shape)
        np.savez_compressed(fn + ".tmp.npz", tumour=mask_from_multilabel(v, lv, "liver_tumor"), liver=mask_from_multilabel(L, tot, "liver")); os.replace(fn + ".tmp.npz", fn); n += 1
        print(f"[ts] {i}/{len(cases)} {c} tumour voxels {int(np.load(fn)['tumour'].sum())} elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"DONE ts: {n} new series of {len(cases)}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
