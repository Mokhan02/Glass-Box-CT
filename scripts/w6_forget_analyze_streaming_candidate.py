"""[CANDIDATE, memory-only change of the frozen scripts/w6_forget_analyze.py: the drift part streams one series at a time; same definitions and arithmetic; NOT in use until the researcher signs it off; equality with the frozen script is tested on real data] W6 forgetting-guard analysis (results/w6_vista_finetune/PREREG.md section 7; operational definitions fixed in Amendment 2 before any fine-tuned output existed). CPU. Frozen and hashed before any arm output; later changes are logged deviations.
Inputs per model: the evaluation folder of scripts/w6_forget_eval.py (spleen.csv, kidney.csv, kidney_tumour_side_check.csv, liver.csv, drift/*.npz, drift_meta.json) and, for the false-positive measure, the W4-route WAW-TACE prediction folders (npz with `tumour`) of the model.
Criteria (each arm against the BASELINE, the same cases):
 (1) organ Dice (spleen, kidneys, liver): the mean paired change (arm - baseline); PASS if the mean change >= -0.02 AND the 95% case-level bootstrap CI lower bound of the change > -0.03.
 (2) prediction drift of all classes except 0 and 26 on the 20 fixed WAW-TACE series: per class the Dice between the baseline's and the arm's mask per series; classes with a median baseline volume (over the 20 series) of at least 10 mL are ABOVE THE FLOOR; PASS if EVERY class above the floor has a mean drift Dice >= 0.95
     (series where both masks are empty count as Dice 1); classes below the floor are reported descriptively with their volumes.
 (3) false positives: predicted class-26 voxels farther than 10 mm from every ground-truth tumour voxel of the series (WAW-TACE, all 396 series), in mL per series; PASS if the arm's median is <= 1.25 x the baseline's median + 0.5 mL (the 0.5 mL tolerance keeps a baseline median of 0 from making the rule impossible).
 Side check, reported only (no pass criterion): the W2 kidney-tumour Dice (point route, class 133).
Result: GUARD PASS (all three) or the list of failed measures; an arm that closes the gap and fails is reported as "fix with forgetting cost" naming them.
Usage: python scripts/w6_forget_analyze.py <out json> --base <eval dir> --arm NAME=<eval dir> [...] [--waw <w4_waw root>] [--pred NAME=<dir1,dir2> ...] (--pred BASELINE=... for the baseline's own predictions)"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

FLOOR_ML, DRIFT_MIN, DROP_MIN, CI_MIN, FP_RATIO, FP_TOL_ML = 10.0, 0.95, -0.02, -0.03, 1.25, 0.5


def read(path, col):
    return {r["case"] if "rank" not in r else f"{r['case']}#{r['rank']}": float(r[col]) for r in csv.DictReader(open(path))}


def paired_change(base, arm, B=2000, name="w6_forget"):
    ks = sorted(set(base) & set(arm)); d = np.array([arm[k] - base[k] for k in ks]); rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)); bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)); m = float(d.mean())
    return {"n": len(ks), "base_mean": float(np.mean([base[k] for k in ks])), "arm_mean": float(np.mean([arm[k] for k in ks])), "change": m, "lo": lo, "hi": hi, "pass": bool(m >= DROP_MIN and lo > CI_MIN)}


def drift_summary(base_masks, arm_masks, volumes_ml):
    """base_masks / arm_masks: {class: [mask per series]}; volumes_ml: {class: [baseline volume in mL per series]}. Returns per-class mean Dice, floor membership and the PASS flag."""
    dice = {}
    for c in base_masks:
        ds = []
        for a, b in zip(base_masks[c], arm_masks[c]):
            s = a.sum() + b.sum(); ds.append(1.0 if s == 0 else 2.0 * (a & b).sum() / s)
        dice[c] = ds
    return summarize_drift(dice, volumes_ml)


def summarize_drift(dice, volumes_ml):
    """Same output as the original drift_summary; `dice` = {class: [Dice per series]}, so the masks need not be held in memory."""
    out = {}
    for c in dice:
        v = float(np.median(volumes_ml[c])); out[c] = {"mean_dice": float(np.mean(dice[c])), "median_baseline_volume_ml": v, "above_floor": v >= FLOOR_ML}
    fl = {c: r for c, r in out.items() if r["above_floor"]}
    return {"classes": out, "n_above_floor": len(fl), "failing_above_floor": sorted((c for c, r in fl.items() if r["mean_dice"] < DRIFT_MIN), key=int), "pass": bool(all(r["mean_dice"] >= DRIFT_MIN for r in fl.values()))}


def fp_ml(pred, gt_tumour, spacing, dist_mm=10.0):
    """Predicted voxels farther than dist_mm from every ground-truth tumour voxel, in mL."""
    from scipy import ndimage
    if not pred.any():
        return 0.0
    far = ndimage.distance_transform_edt(~gt_tumour, sampling=spacing) > dist_mm if gt_tumour.any() else np.ones_like(pred)
    return float((pred & far).sum() * np.prod(spacing) / 1000.0)


def load_drift(dirp):
    meta = json.load(open(os.path.join(dirp, "drift_meta.json"))); return meta


def main(argv):
    out = argv[0]; arms = {}; base = None; preds = {}; waw = None
    for i, a in enumerate(argv):
        if a == "--base":
            base = argv[i + 1]
        elif a == "--arm":
            n, p = argv[i + 1].split("=", 1); arms[n] = p
        elif a == "--pred":
            n, p = argv[i + 1].split("=", 1); preds[n] = p.split(",")
        elif a == "--waw":
            waw = argv[i + 1]
    res = {}
    for n, d in arms.items():
        r = {"organ": {}, "drift": None, "false_positives": None, "side_check_kidney_tumour": None}
        for nm, f, col in (("spleen", "spleen.csv", "dice"), ("kidneys", "kidney.csv", "dice"), ("liver", "liver.csv", "liver_dice")):
            if os.path.exists(os.path.join(base, f)) and os.path.exists(os.path.join(d, f)):
                r["organ"][nm] = paired_change(read(os.path.join(base, f), col), read(os.path.join(d, f), col), name=f"w6_forget_{n}_{nm}")
        sc = "kidney_tumour_side_check.csv"
        if os.path.exists(os.path.join(base, sc)) and os.path.exists(os.path.join(d, sc)):
            r["side_check_kidney_tumour"] = paired_change(read(os.path.join(base, sc), "dice"), read(os.path.join(d, sc), "dice"), name=f"w6_forget_{n}_kt")
        if os.path.isdir(os.path.join(base, "drift")) and os.path.isdir(os.path.join(d, "drift")):
            meta = load_drift(base); dice = {str(c): [] for c in meta["classes"]}; vol = {str(c): [] for c in meta["classes"]}
            for s_ in meta["series"]:      # one series at a time: the masks of the whole cohort are never held in memory
                fn = s_.replace("/", "_") + ".npz"; zb, za = np.load(os.path.join(base, "drift", fn)), np.load(os.path.join(d, "drift", fn)); v = float(np.prod(zb["spacing"])) / 1000.0
                for c in dice:
                    a, b = zb[c], za[c]; t = a.sum() + b.sum(); dice[c].append(1.0 if t == 0 else 2.0 * (a & b).sum() / t); vol[c].append(float(a.sum()) * v)
            r["drift"] = summarize_drift(dice, vol)
        if waw and "BASELINE" in preds and n in preds:
            from nibabel import load as nload

            def per_series(dirs):
                v = {}
                for dp in dirs:
                    half = "half0" if "half0" in dp else "half1"
                    for f in sorted(os.listdir(dp)):
                        if f.endswith(".npz") and f[0].isdigit():
                            c = f[:-4]; g = os.path.join(waw, half, "gt", c); lab = np.asarray(nload(os.path.join(g, "tumour_labels.nii.gz")).dataobj) > 0; sp = tuple(float(z) for z in nload(os.path.join(g, "ct.nii.gz")).header.get_zooms()[:3])
                            v[f"{half}/{c}"] = fp_ml(np.load(os.path.join(dp, f))["tumour"].astype(bool), lab, sp)
                return v
            fb, fa = per_series(preds["BASELINE"]), per_series(preds[n]); ks = sorted(set(fb) & set(fa)); mb, ma = float(np.median([fb[k] for k in ks])), float(np.median([fa[k] for k in ks]))
            r["false_positives"] = {"n_series": len(ks), "median_ml_baseline": mb, "median_ml_arm": ma, "mean_ml_baseline": float(np.mean([fb[k] for k in ks])), "mean_ml_arm": float(np.mean([fa[k] for k in ks])), "pass": bool(ma <= FP_RATIO * mb + FP_TOL_ML)}
        passes = {**{f"organ_{k}": v["pass"] for k, v in r["organ"].items()}, **({"drift": r["drift"]["pass"]} if r["drift"] else {}), **({"false_positives": r["false_positives"]["pass"]} if r["false_positives"] else {})}
        r["measures"] = passes; r["failed"] = sorted(k for k, v in passes.items() if not v); r["guard"] = "GUARD PASS" if passes and not r["failed"] else ("GUARD INCOMPLETE" if not passes else "FORGETTING COST: " + ", ".join(r["failed"])); res[n] = r
    json.dump(res, open(out, "w"), indent=1, default=float); print(json.dumps({n: {"guard": r["guard"], "failed": r["failed"]} for n, r in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
