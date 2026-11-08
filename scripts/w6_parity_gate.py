"""W6 environment parity gate (PREREG W6, section 6, amendment 1): BEFORE any arm trains, the training environment must reproduce the W4 baseline class-26 predictions on 3 fixed WAW-TACE series: byte-identical to the stored W4 mask, or Dice >= 0.99 (class 1, the liver, is reported as information).
The route is exactly the W4 route (`glassbox.seg.w2_lib.Runner`: `prep`, `predict_auto(data, img, 26)`: automatic class prompt, whole-volume sliding window 128^3, overlap 0.3, probability > 0.5). The 3 series: a seeded sample (seed = first 8 hex digits of the SHA-256 of `w6_parity`) among the series whose stored W4 class-26 mask has at least 1,000 voxels (so that the Dice is not trivial).
Exit code 1 (and `"pass": false`) if any series fails: stop and report. Usage (python of the NEW environment, GPU, cwd = repo checkout): python scripts/w6_parity_gate.py <vendored_inferer_dir> <weights> <w4 root with half0/ half1/ (gt/, pred_vista/)> <out.json> [--n 3] [--dice-min 0.99]"""
import hashlib
import json
import os
import random
import sys

import numpy as np

MIN_VOXELS = 1000


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def choose_cases(candidates, n=3):
    """candidates: sorted list of (half, case, n_voxels_of_the_stored_class_26_mask); returns n of those with at least MIN_VOXELS, seeded."""
    ok = sorted(c for c in candidates if c[2] >= MIN_VOXELS); rng = random.Random(int(hashlib.sha256(b"w6_parity").hexdigest()[:8], 16)); return sorted(rng.sample(ok, n))


def compare(new, stored, dice_min=0.99):
    same = bool(np.array_equal(np.asarray(new, bool), np.asarray(stored, bool))); d = dice(new, stored)
    return {"byte_identical": same, "dice": d, "pass": bool(same or d >= dice_min), "voxels_new": int(np.asarray(new).sum()), "voxels_stored": int(np.asarray(stored).sum())}


def main(argv):
    inf, wts, root, out = argv[:4]; n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 3; dmin = float(argv[argv.index("--dice-min") + 1]) if "--dice-min" in argv else 0.99
    import monai
    import torch
    sys.path.insert(0, "."); from glassbox.seg.w2_lib import Runner      # the W4 route
    cands = []
    for h in ("half0", "half1"):
        gt = os.path.join(root, h, "gt"); pr = os.path.join(root, h, "pred_vista")
        for c in sorted(d for d in os.listdir(gt) if d[0].isdigit()):
            f = os.path.join(pr, c + ".npz")
            if os.path.exists(f):
                cands.append((h, c, int(np.load(f)["tumour"].sum())))
    sel = choose_cases(cands, n); R = Runner(inf, wts); res = []
    for h, c, nv in sel:
        data, img, _ = R.prep(os.path.join(root, h, "gt", c, "ct.nii.gz")); tum = np.asarray(R.predict_auto(data, img, 26)).astype(bool); liv = np.asarray(R.predict_auto(data, img, 1)).astype(bool); z = np.load(os.path.join(root, h, "pred_vista", c + ".npz"))
        r = {"half": h, "case": c, "class26": compare(tum, z["tumour"], dmin), "class1_info": compare(liv, z["liver"], dmin)}; res.append(r); print(c, r["class26"], flush=True)
    rep = {"pass": all(r["class26"]["pass"] for r in res), "dice_min": dmin, "cases": res, "monai": monai.__version__, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "n_candidates": len(cands)}
    try:
        import ignite; rep["ignite"] = ignite.__version__
    except ImportError:
        rep["ignite"] = None
    json.dump(rep, open(out, "w"), indent=1); print("PARITY GATE", "PASS" if rep["pass"] else "FAIL"); sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
