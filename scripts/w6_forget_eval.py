"""W6 forgetting-guard inference (results/w6_vista_finetune/PREREG.md section 7), on one model (the baseline or a fine-tuned arm), GPU. Same code and the same cases for every model; fixed case lists (seeds = first 8 hex digits of the SHA-256 of the names below).
Tasks: spleen (MSD Task09, all labelled cases, class 3 vs label 1); kidney (C4KC-KiTS, 60 fixed cases, classes 5 | 14 vs the ground-truth kidney = seg > 0; and the W2 kidney-tumour side check: class 133 with one positive point of label 1 at each primary lesion's interior point, as in W2);
liver (MSD Task03 nnU-Net fold-0 validation cases: class 1 vs labels 1 | 2, and the automatic class 26 vs label 2, descriptive); drift (20 fixed WAW-TACE series: every class of the bundle's label_dict except 0 and 26, one binary mask per class saved compressed, to be compared between models by scripts/w6_forget_analyze.py).
Usage (python of the W6 environment, GPU, cwd = repo checkout): python scripts/w6_forget_eval.py <inferer_dir> <weights> <arm> <out dir> --spleen <Task09 dir> --kits <c4kc_kits dir> --task03 <Task03 dir> --datalist <task03 datalist.json> --waw <w4_waw root> --labels <label_dict.json> [--tasks spleen,kidney,liver,drift] [--chunk 66]"""
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np


def seeded_sample(items, n, name):
    items = sorted(items); rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)); return sorted(rng.sample(items, min(n, len(items))))


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); s = a.sum() + b.sum(); return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def drift_classes(label_dict):
    """All class indices of the bundle's label_dict except 0 and 26 (26 is the class the fine-tune is meant to change)."""
    return sorted(int(k) for k in label_dict if int(k) not in (0, 26))


def main(argv):
    inf, wts, arm, out = argv[:4]; opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    tasks = (opt("--tasks", "spleen,kidney,liver,drift")).split(","); chunk = int(opt("--chunk", 66)); os.makedirs(out, exist_ok=True)
    import nibabel as nib
    import torch
    sys.path.insert(0, "."); from glassbox.seg import infer as I
    from glassbox.seg.w2_lib import Runner, case_geometry, interior_point
    R = Runner(inf, wts); t0 = time.time(); log = lambda m: print(f"[forget {arm}] {m} ({time.time() - t0:.0f}s)", flush=True)      # noqa: E731
    def prep(path):
        d, im, M = R.prep(path); return d, im, M
    if "spleen" in tasks:
        root = opt("--spleen"); cases = sorted(f for f in os.listdir(os.path.join(root, "imagesTr")) if f.endswith(".nii.gz") and not f.startswith("._")); rows = []
        for f in cases:
            d, im, _ = prep(os.path.join(root, "imagesTr", f)); gt = np.asarray(nib.load(os.path.join(root, "labelsTr", f)).dataobj) == 1; pr = np.asarray(R.predict_auto(d, im, 3)).astype(bool)[:gt.shape[0], :gt.shape[1], :gt.shape[2]]
            rows.append({"case": f, "dice": dice(pr, gt), "pred_voxels": int(pr.sum()), "gt_voxels": int(gt.sum())})
        with open(os.path.join(out, "spleen.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        log(f"spleen {len(rows)} cases, mean dice {np.mean([r['dice'] for r in rows]):.4f}")
    if "kidney" in tasks:
        root = opt("--kits"); cases = seeded_sample([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], 60, "w6_kits"); krows, trows = [], []
        for c in cases:
            nii, sp, seg, ann, zidx, lab, sel = case_geometry(root, c); d, im, M = prep(os.path.join(root, c, "ct.nii.gz")); sh = seg.shape
            k = (np.asarray(R.predict_auto(d, im, 5)).astype(bool) | np.asarray(R.predict_auto(d, im, 14)).astype(bool))[:sh[0], :sh[1], :sh[2]]; krows.append({"case": c, "dice": dice(k, seg > 0)})
            for rank, (diam, cid) in enumerate(sel[:2], 1):
                comp_c = lab == cid; pc = interior_point(comp_c, sp); comp = np.zeros(sh, bool); comp[:, :, ann] = comp_c; pt = (pc[0], pc[1], int(zidx[pc[2]]))
                pr = np.asarray(R.predict(d, im, M, 133, 1, pt)).astype(bool)[:sh[0], :sh[1], :sh[2]]; trows.append({"case": c, "rank": rank, "diam_mm": diam, "dice": dice(pr, comp)})
        for nm, rows in (("kidney.csv", krows), ("kidney_tumour_side_check.csv", trows)):
            with open(os.path.join(out, nm), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        log(f"kidney {len(krows)} cases mean dice {np.mean([r['dice'] for r in krows]):.4f}; tumour side check {len(trows)} lesions mean dice {np.mean([r['dice'] for r in trows]):.4f}")
    if "liver" in tasks:
        root = opt("--task03"); rws = [r for r in json.load(open(opt("--datalist")))["training"] if r["fold"] == 0]; rows = []
        for r in rws:
            lab = np.asarray(nib.load(os.path.join(root, r["label"])).dataobj); d, im, _ = prep(os.path.join(root, r["image"])); sh = lab.shape
            liv = np.asarray(R.predict_auto(d, im, 1)).astype(bool)[:sh[0], :sh[1], :sh[2]]; tum = np.asarray(R.predict_auto(d, im, 26)).astype(bool)[:sh[0], :sh[1], :sh[2]]
            rows.append({"case": r["image"], "liver_dice": dice(liv, lab > 0), "tumour_dice_descriptive": dice(tum, lab == 2), "has_tumour": int((lab == 2).any()), "pred_tumour_voxels": int(tum.sum())})
        with open(os.path.join(out, "liver.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        log(f"liver {len(rows)} validation cases mean dice {np.mean([r['liver_dice'] for r in rows]):.4f}")
    if "drift" in tasks:
        waw = opt("--waw"); series = [(h, c) for h in ("half0", "half1") for c in sorted(d for d in os.listdir(os.path.join(waw, h, "gt")) if d[0].isdigit())]
        pick = seeded_sample([f"{h}/{c}" for h, c in series], 20, "w6_drift"); ids = drift_classes(json.load(open(opt("--labels")))); os.makedirs(os.path.join(out, "drift"), exist_ok=True); meta = {"classes": ids, "series": pick}
        for s in pick:
            h, c = s.split("/"); path = os.path.join(waw, h, "gt", c, "ct.nii.gz"); fn = os.path.join(out, "drift", f"{h}_{c}.npz")
            if os.path.exists(fn):
                continue
            masks = {}; sp = tuple(float(z) for z in nib.load(path).header.get_zooms()[:3])
            for i in range(0, len(ids), chunk):
                cls = ids[i:i + chunk]; probs, _ = I.predict_probs(path, cls, R.net, R.dev)
                for j, k in enumerate(cls):
                    m = (probs[j] > 0.5).numpy(); masks[str(k)] = m
                del probs; torch.cuda.empty_cache()
            np.savez_compressed(fn, spacing=np.array(sp), **masks); log(f"drift {s} done")
        json.dump(meta, open(os.path.join(out, "drift_meta.json"), "w"))
    json.dump({"arm": arm, "weights": os.path.basename(wts), "tasks": tasks, "wall_seconds": time.time() - t0}, open(os.path.join(out, "forget_eval_done.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
