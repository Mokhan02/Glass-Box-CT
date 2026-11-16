"""[EXPLORATORY] Validation of the TotalSegmentator --fast pseudo-liver against the PREREG full-resolution one on 5 seeded Task08 cases (results/w1_training_audit/DEVIATIONS.md, entry of 2026-11-15, rule fixed before the run): liver Dice (largest component of each) and the lesion contrast with each mask; max |delta c| and bright/dark flips. Both masks must already exist: <out>/pseudo_liver (full) and <out>/pseudo_liver_fast.
Usage (cwd = repo root, venv with nibabel/scipy): python scripts/w1_task08_validate.py <Task08 dir> <out dir>      -> <out>/validation.json and a printed summary"""
import json
import os
import random
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from w1_task08_extract import case_rows, names  # noqa: E402


def lcc(m):
    cc, n = ndimage.label(m); return cc == (1 + int(np.argmax(np.bincount(cc.ravel())[1:]))) if n else m


def main(root, out):
    picks = random.Random(20261115).sample(names(root), 5); res = {"cases": picks, "per_case": {}}
    for n in picks:
        full, fast = os.path.join(out, "pseudo_liver", n), os.path.join(out, "pseudo_liver_fast", n)
        a = lcc(np.asarray(nib.load(full).dataobj) > 0); b = lcc(np.asarray(nib.load(fast).dataobj) > 0); dice = float(2 * (a & b).sum() / (a.sum() + b.sum()))
        rf, _ = case_rows(root, n, full); rq, _ = case_rows(root, n, fast); les = []
        for x, y in zip(rf, rq):
            les.append({"rank": x["rank"], "diam_mm": x["diam_mm"], "c_full": x["contrast_hu"], "c_fast": y["contrast_hu"], "dc": y["contrast_hu"] - x["contrast_hu"], "flip": (x["contrast_hu"] > 0) != (y["contrast_hu"] > 0), "fallback_full": x["fallback"], "fallback_fast": y["fallback"]})
        res["per_case"][n] = {"dice": dice, "n_lesions": len(les), "lesions": les}
    allL = [l for c in res["per_case"].values() for l in c["lesions"]]
    res["summary"] = {"min_dice": min(c["dice"] for c in res["per_case"].values()), "n_lesions": len(allL), "max_abs_dc_hu": max((abs(l["dc"]) for l in allL), default=float("nan")), "median_abs_dc_hu": float(np.median([abs(l["dc"]) for l in allL])) if allL else float("nan"),
                      "n_flips": int(sum(l["flip"] for l in allL)), "all_dice_ge_0.95": all(c["dice"] >= 0.95 for c in res["per_case"].values())}
    res["summary"]["PASS"] = bool(res["summary"]["all_dice_ge_0.95"] and res["summary"]["n_flips"] == 0)
    json.dump(res, open(os.path.join(out, "validation.json"), "w"), indent=1)
    for n, c in res["per_case"].items():
        print(n, f"dice {c['dice']:.4f} lesions {c['n_lesions']}", [f"{l['c_full']:+.1f}->{l['c_fast']:+.1f}" for l in c["lesions"]][:6])
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
