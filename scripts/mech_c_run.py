"""Mechanistic experiment C, step 2 (results/mech/PREREG.md section C): steering at the mid-encoder and bottleneck. GPU.
Phase select : per fold, 20 random training cases x alpha grid x layer, probe direction -> case Dice and healthy-liver false-positive volume.
Phase heldout: per fold, alpha (max mean Dice on the 20 cases) and matched threshold t (matched FP volume on the same 20 cases); then on held-out cases: original, threshold, probe, random, mean-difference arms.
Step added at every feature vector in the ROI cells (predicted liver + 10 mm, cell fraction >= 0.5): alpha * dmu * direction. Ground truth is used only for scoring.
Usage: python scripts/mech_c_run.py --phase select|heldout --shard 0 --nshards 2"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from monai.inferers import SlidingWindowInfererAdapt
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_transforms  # noqa: E402
from glassbox.extract.hooks import window_pads  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["select", "heldout"]); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", default="cache/mech_c"); ap.add_argument("--rr", default="cache/rr"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
K, VOX, SUB = 5, 1.5, 20
ALPHAS = [0.25, 0.5, 1.0, 1.5]; TGRID = [-0.25 * i for i in range(21)]; LAYERS = {"mid": (2, 4), "bottleneck": (4, 16)}
dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev)
D = np.load(os.path.join(a.out, "dirs.npz"))
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(cases)}
subset = {f: sorted(np.random.default_rng(f).choice([c for c in cases if fold_of[c] != f], SUB, replace=False).tolist()) for f in range(K)}
st = {"layer": None, "step": None, "roi": None, "p": -1, "coords": []}


def _pre(module, args_, kwargs):
    slices = args_[1][0] if len(args_) > 1 else kwargs["patch_coords"][0]
    st["p"] += 1; st["coords"].append(tuple(slices[-3:])); return None


def make_hook(name):
    stride = LAYERS[name][1]

    def hook(module, args_, out):
        if st["layer"] != name:
            return None
        sl = st["coords"][st["p"]]; cm = Fn.avg_pool3d(st["roi"][sl].float()[None, None], stride)[0, 0] >= 0.5
        return out + st["step"][None, :, None, None, None] * cm[None, None].float() if cm.any() else None
    return hook


net.register_forward_pre_hook(_pre, with_kwargs=True)
for n, (idx, _) in LAYERS.items():
    net.image_encoder.encoder.layers[idx]["blocks"].register_forward_hook(make_hook(n))


def feret(m2):
    pts = np.argwhere(m2).astype(float) * VOX
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def lesions(m3):
    lab, n = ndimage.label(m3, structure=np.ones((3, 3, 3))); out = []
    for i in range(1, n + 1):
        out.append({"id": i, "diam": max(feret(lab[:, :, z] == i) for z in np.flatnonzero((lab == i).any(axis=(0, 1))))})
    return lab, out


class Ctx:
    def __init__(self, c):
        z = np.load(os.path.join(a.rr, c + ".npz")); self.c, self.shape, self.route = c, tuple(z["shape"]), str(z["route"]); L = z["L"]; self.contrast = float(z["contrast_hu"])
        self.ann = (L != 3).any(axis=(0, 1)); Lc = L[:, :, self.ann]; self.gt = Lc == 2
        dist = ndimage.distance_transform_edt(~self.gt) * VOX; self.fpmask = (Lc == 1) & (dist > 10.0)
        self.roi_idx = z["roi_idx"].astype(np.int64); ref = z["logit_full"].astype(np.float32).ravel(); ref[self.roi_idx] = z["logit_roi"]; self.ref = ref.reshape(self.shape)
        roi = np.zeros(int(np.prod(self.shape)), bool); roi[self.roi_idx] = True; self.roi = roi.reshape(self.shape)
        self.glab, gl = lesions(self.gt); self.gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]

    def score(self, logit, full):
        pred = (logit[:, :, self.ann] > 0); dice = float(2 * (pred & self.gt).sum() / max(pred.sum() + self.gt.sum(), 1)); fp = float((pred & self.fpmask).sum() * VOX ** 3 / 1000.0)
        out = {"dice": dice, "fp_cm3": fp}
        if full:
            nt = nd = 0; rel = float("nan")
            if self.gl:
                plab, pl = lesions(pred); pd_ = {l["id"]: l["diam"] for l in pl}; sg = sp = 0.0
                for l in self.gl:
                    gm = self.glab == l["id"]; ov = np.unique(plab[gm & (plab > 0)]); det = (pred & gm).sum() / max(gm.sum(), 1) >= 0.10
                    sg += l["diam"]; nd += int(det); sp += max([pd_[o] for o in ov], default=0.0) if det else 0.0
                nt = len(self.gl); rel = (sp - sg) / sg
            out.update({"n_targets": nt, "n_detected": nd, "relerr": rel})
        return out

    def thr_logit(self, t):
        return np.where(self.roi, self.ref - t, self.ref)  # logit > t inside the ROI  <=>  (logit - t) > 0; outside unchanged (threshold 0)

    def image(self):
        return build_transforms()({"image": os.path.join(ROOTS[self.route], self.c, "ct.nii.gz")})["image"].unsqueeze(0).to(dev)


def steer(ctx, x, layer, step_vec):
    before, after = window_pads(ctx.shape)
    roi = Fn.pad(torch.from_numpy(ctx.roi.astype(np.float32)).to(dev), [p for k in (2, 1, 0) for p in (before[k], after[k])]) > 0.5
    st.update({"layer": layer, "step": torch.from_numpy(step_vec.astype(np.float32)).to(dev), "roi": roi, "p": -1, "coords": []})
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    with torch.no_grad():
        lg = sw(x, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[26]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
    st["layer"] = None
    return lg.reshape(*x.shape[2:]).float().cpu().numpy()


def write(path, rows):
    names = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, restval=""); w.writeheader(); w.writerows(rows)


rows, t0 = [], time.time()
if a.phase == "select":
    jobs = [(f, c) for f in range(K) for c in subset[f]][a.shard::a.nshards][: a.limit or None]
    for ji, (f, c) in enumerate(jobs, 1):
        ctx = Ctx(c); x = ctx.image()
        for layer in LAYERS:
            u, dmu = D[f"{layer}_f{f}_u"], float(D[f"{layer}_f{f}_dmu"])
            for al in ALPHAS:
                sc = ctx.score(steer(ctx, x, layer, al * dmu * u), full=False); rows.append({"fold": f, "case": c, "layer": layer, "alpha": al, **{k: round(v, 5) for k, v in sc.items()}})
        write(os.path.join(a.out, f"select_shard{a.shard}.csv"), rows); print(f"[{ji}/{len(jobs)}] fold {f} {c}; {time.time() - t0:.0f}s", flush=True)
else:
    sel = [r for fn in glob.glob(os.path.join(a.out, "select_shard*.csv")) for r in csv.DictReader(open(fn))]
    alpha, tmatch = {}, {}
    for f in range(K):
        for layer in LAYERS:
            m = {al: np.mean([float(r["dice"]) for r in sel if int(r["fold"]) == f and r["layer"] == layer and float(r["alpha"]) == al]) for al in ALPHAS}
            alpha[(f, layer)] = max(m, key=m.get)
            fp_s = np.mean([float(r["fp_cm3"]) for r in sel if int(r["fold"]) == f and r["layer"] == layer and float(r["alpha"]) == alpha[(f, layer)]])
            tmatch[(f, layer)] = fp_s
    # thresholds: matched healthy-liver FP volume on the same 20 cases (computed once per fold)
    tsel = {}
    for f in range(K):
        ctxs = [Ctx(c) for c in subset[f]]
        fp_t = [np.mean([cx.score(cx.thr_logit(t), full=False)["fp_cm3"] for cx in ctxs]) for t in TGRID]
        for layer in LAYERS:
            tsel[(f, layer)] = TGRID[int(np.argmin(np.abs(np.array(fp_t) - tmatch[(f, layer)])))]
        print("fold", f, {l: (alpha[(f, l)], tsel[(f, l)]) for l in LAYERS}, flush=True)
    import json
    json.dump({f"{f}|{l}": {"alpha": alpha[(f, l)], "t": tsel[(f, l)], "steer_fp_cm3_on_subset": tmatch[(f, l)]} for f in range(K) for l in LAYERS}, open(os.path.join(a.out, "chosen.json"), "w"), indent=1)
    jobs = cases[a.shard::a.nshards][: a.limit or None]
    for ji, c in enumerate(jobs, 1):
        f = fold_of[c]; ctx = Ctx(c); x = ctx.image(); base = {"case": c, "fold": f, "contrast_hu": ctx.contrast}
        rows.append({**base, "layer": "-", "arm": "orig", **{k: round(v, 5) for k, v in ctx.score(ctx.ref, True).items()}})
        for layer in LAYERS:
            al, dmu = alpha[(f, layer)], float(D[f"{layer}_f{f}_dmu"]); C = D[f"{layer}_f{f}_u"].shape[0]
            rows.append({**base, "layer": layer, "arm": "thr", "alpha": "", "t": tsel[(f, layer)], **{k: round(v, 5) for k, v in ctx.score(ctx.thr_logit(tsel[(f, layer)]), True).items()}})
            rv = np.random.default_rng([f, int(c.split("_")[1])]).normal(size=C); rv /= np.linalg.norm(rv)
            for arm, vec in (("probe", D[f"{layer}_f{f}_u"]), ("random", rv), ("meandiff", D[f"{layer}_f{f}_md"])):
                rows.append({**base, "layer": layer, "arm": arm, "alpha": al, "t": "", **{k: round(v, 5) for k, v in ctx.score(steer(ctx, x, layer, al * dmu * vec), True).items()}})
        write(os.path.join(a.out, f"heldout_shard{a.shard}.csv"), rows); print(f"[{ji}/{len(jobs)}] {c}; {time.time() - t0:.0f}s", flush=True)
