"""Mechanistic experiment D: counterfactual patching on the 62 Arm A lesions (results/mech/PREREG.md, section D). GPU.
DARK = unedited scan; BRIGHT = the pre-registered edit (same edit_lesion + target as results/contrast_cf). Run BRIGHT with DARK activations patched in, per layer and region.
Layers: mid = encoder.layers[2]["blocks"], bottleneck = encoder.layers[4]["blocks"], decoder_F = class_head.image_post_mapping. Regions: R1 dilated tumour (<=10 mm), R2 shell (0<d<=10 mm), R3 far (>10 mm),
control = R1 filled with the DARK activations of an unrelated pool case at the same voxel indices.
Usage: python scripts/mech_d_patch.py --shard 0 --nshards 3 --out cache/mech_d"""
import argparse
import csv
import os
import sys
import time
from functools import partial

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as Fn
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import Invertd, LoadImage
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.bodycomp.counterfactual import edit_lesion  # noqa: E402
from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_pair_transforms, build_transforms  # noqa: E402
from glassbox.extract.hooks import window_pads  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", default="cache/mech_d"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "scratch"), exist_ok=True)
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
STRIDE = {"mid": 4, "bottleneck": 16, "decoder_F": 1}
LAYERS = ["mid", "bottleneck", "decoder_F"]
dev = torch.device("cuda")
net = load_network("cache/weights/model.pt", dev)
load = LoadImage(image_only=True)
VOX = 1.5


class Hooker:
    """Records window activations (mode 'record') or patches them (mode 'patch') at one layer."""
    def __init__(self, net):
        self.mode, self.p, self.coords, self.store, self.rec_layers = None, -1, [], {}, ()
        self.patch_layer, self.region, self.before, self.src = None, None, None, None  # src(name, p, sl) -> (tensor on device, valid bool [g,g,g]) or None
        mods = {"mid": net.image_encoder.encoder.layers[2]["blocks"], "bottleneck": net.image_encoder.encoder.layers[4]["blocks"], "decoder_F": net.class_head.image_post_mapping}
        net.register_forward_pre_hook(self._pre, with_kwargs=True)
        for n, m in mods.items():
            m.register_forward_hook(partial(self._post, n))

    def reset(self, mode=None):
        self.mode, self.p, self.coords, self.store = mode, -1, [], {n: [] for n in LAYERS}

    def _pre(self, module, args, kwargs):
        slices = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]
        self.p += 1; self.coords.append(tuple(slices[-3:])); return None

    def _post(self, name, module, args, out):
        if self.mode == "record" and name in self.rec_layers:
            self.store[name].append(out.detach().to("cpu", torch.float16))
        elif self.mode == "patch" and name == self.patch_layer:
            sl = self.coords[self.p]; st = STRIDE[name]
            w = self.region[sl].float()[None, None]
            cm = (Fn.avg_pool3d(w, st) if st > 1 else w)[0, 0] >= 0.5
            if not cm.any():
                return None
            got = self.src(name, self.p, sl)
            if got is None:
                return None
            act, valid = got; cm = cm & valid.to(cm.device)
            return torch.where(cm[None, None], act.to(out.dtype), out) if cm.any() else None


hk = Hooker(net)


def infer(pre, data, want_inv=True):
    img = data["image"].unsqueeze(0).to(dev); sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    with torch.no_grad():
        lg = sw(img, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[26]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
    lg = lg.reshape(*img.shape[2:]).float()
    if not want_inv:
        return lg.cpu(), None
    p = MetaTensor(torch.sigmoid(lg).cpu()[None], meta=data["image"].meta, applied_operations=data["image"].applied_operations)
    out = Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True)({"pred": p, "image": data["image"]})["pred"].as_tensor()[0].numpy() > 0.5
    return lg.cpu(), out


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def assemble(store, coords, before, shape):
    """Window-average the recorded activations of one layer into a full layer-grid volume (padded coordinates), fp16, CPU."""
    out = {}
    for name in LAYERS:
        st = STRIDE[name]; C = store[name][0].shape[1]; g = PATCH_SIZE[0] // st
        pshape = [max(c[k].stop for c in coords) // st for k in range(3)]
        vol = torch.zeros(C, *pshape); cnt = torch.zeros(*pshape)
        for p, act in enumerate(store[name]):
            s0 = [s.start // st for s in coords[p]]
            sl = tuple(slice(s0[k], s0[k] + g) for k in range(3))
            sub = vol[(slice(None),) + sl]; gshape = sub.shape[1:]
            vol[(slice(None),) + sl] += act[0, :, : gshape[0], : gshape[1], : gshape[2]].float(); cnt[sl] += 1
        out[name] = (vol / cnt.clamp(min=1)).half()
    return out


# ---- the 62 Arm A lesions (rows of the counterfactual)
edits = [r for r in csv.DictReader(open("cache/contrast_cf/edits.csv")) if r["arm"] == "A" and r["cond"] == "edit"]
edits = sorted(edits, key=lambda r: (r["case"], int(r["rank"])))
cases_all = sorted({r["case"] for r in edits})
my_cases = [c for i, c in enumerate(cases_all) if i % a.nshards == a.shard][: a.limit or None]
src_of = {r["case"]: r["source"] for r in edits}
POOL = cases_all[:5]


def case_geometry(case):
    root = ROOTS[src_of[case]]; nii = nib.load(os.path.join(root, case, "ct.nii.gz")); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    return root, nii, seg


def dark_record(case):
    """Unedited run with all three layers recorded. Returns pre, data, coords, store, before, shape, logit, pred."""
    root = ROOTS[src_of[case]]; path = os.path.join(root, case, "ct.nii.gz"); pre = build_transforms(); data = pre({"image": path})
    shape = tuple(data["image"].shape[1:]); before, _ = window_pads(shape)
    hk.reset("record"); hk.rec_layers = tuple(LAYERS); lg, pred = infer(pre, data); hk.rec_layers = ()
    return pre, data, list(hk.coords), hk.store, before, shape, lg, pred


print("building donor pool", POOL, flush=True)
donors = {}
for c in POOL:
    pre, data, coords, store, before, shape, lg, pred = dark_record(c)
    donors[c] = assemble(store, coords, before, shape); del store
    print("donor", c, {n: tuple(v.shape) for n, v in donors[c].items()}, flush=True)

out_csv = os.path.join(a.out, f"patch_shard{a.shard}.csv"); out_rows = []
t0 = time.time()
for ci, case in enumerate(my_cases, 1):
    root, nii, seg = case_geometry(case); sx, sy, sz = (float(z) for z in nii.header.get_zooms()[:3]); ct = np.asarray(nii.dataobj).astype(np.float32)
    ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann); gt_c = (seg == 2)[:, :, ann]
    lab, n = ndimage.label(gt_c, structure=np.ones((3, 3, 3))); les = []
    for i in range(1, n + 1):
        comp = lab == i; zs_ = np.flatnonzero(comp.any(axis=(0, 1))); les.append((max(feret(comp[:, :, z], sx, sy) for z in zs_), i))
    les = sorted([l for l in les if l[0] >= 10.0], key=lambda l: -l[0])[:2]
    pre, data, coords, store, before, shape, lg_dark, pred_dark = dark_record(case)
    img_meta = load(os.path.join(root, case, "ct.nii.gz"))
    for e in [r for r in edits if r["case"] == case]:
        rank = int(e["rank"]); diam, cid = les[rank - 1]
        xs, ys, ks = np.nonzero(lab == cid); zs = zidx[ks]; X, Y, Z = ct.shape; pad = int(25 / min(sx, sy, sz))
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X); y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y); z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
        box = (slice(x0, x1), slice(y0, y1), slice(z0, z1)); les3 = np.zeros((x1 - x0, y1 - y0, z1 - z0), bool); les3[xs - x0, ys - y0, zs - z0] = True
        dist = ndimage.distance_transform_edt(~les3, sampling=(sx, sy, sz))

        def measure(pred):
            cov = float(pred[xs, ys, zs].mean()); P = pred[box] & (dist <= 10.0); return cov, float(2 * (P & les3).sum() / max(P.sum() + les3.sum(), 1))
        m = np.zeros(ct.shape, bool); m[xs, ys, zs] = True
        edited, info = edit_lesion(ct, m, float(e["target_mean"])); tmp = os.path.join(a.out, "scratch", f"{case}_L{rank}.nii.gz"); nib.save(nib.Nifti1Image(edited, nii.affine), tmp)
        pre_b = build_transforms(); data_b = pre_b({"image": tmp}); assert tuple(data_b["image"].shape[1:]) == shape, "edit changed the preprocessed shape"
        pair = build_pair_transforms()({"image": img_meta, "label": MetaTensor(torch.from_numpy(m.astype(np.int16)), meta=img_meta.meta.copy(), affine=img_meta.affine)})
        core = pair["label"][0].as_tensor().numpy() > 0; assert core.shape == shape
        dpre = ndimage.distance_transform_edt(~core, sampling=(VOX,) * 3)
        pad_t = lambda v: Fn.pad(torch.from_numpy(v.astype(np.float32)).to(dev), [p for k in (2, 1, 0) for p in (before[k], window_pads(shape)[1][k])]) > 0.5  # noqa: E731
        regions = {"R1": pad_t(dpre <= 10.0), "R2": pad_t((dpre > 0) & (dpre <= 10.0)), "R3": pad_t(dpre > 10.0)}
        coreT = torch.from_numpy(core)
        base = {}
        for cond, (pp, dd) in (("dark", (pre, data)), ("bright", (pre_b, data_b))):
            hk.reset(None); lg, pred = infer(pp, dd) if cond == "bright" else (lg_dark, pred_dark)
            cov, dice = measure(pred); base[cond] = (cov, dice, float(lg[coreT].mean()))
            out_rows.append({"case": case, "rank": rank, "layer": "", "region": "", "cond": cond, "cov": round(cov, 4), "dice": round(dice, 4), "logit_core": round(base[cond][2], 3), "n_windows": len(coords)})
        d_ref = abs(base["dark"][0] - float(e["cov0"]))

        def run_patch(layer, region, source):
            def src(name, p, sl):
                if source == "dark":
                    return store[name][p].to(dev)[0].unsqueeze(0), torch.ones((PATCH_SIZE[0] // STRIDE[name],) * 3, dtype=torch.bool)
                dvol = donors[source][name]; st = STRIDE[name]; g = PATCH_SIZE[0] // st; s0 = [s.start // st for s in sl]
                buf = torch.zeros(1, dvol.shape[0], g, g, g, dtype=torch.float16); valid = torch.zeros(g, g, g, dtype=torch.bool); rng = []
                for k in range(3):
                    lo, hi = s0[k], min(s0[k] + g, dvol.shape[1 + k]); rng.append((lo, hi))
                if any(h <= l for l, h in rng):
                    return None
                buf[0, :, : rng[0][1] - rng[0][0], : rng[1][1] - rng[1][0], : rng[2][1] - rng[2][0]] = dvol[:, rng[0][0]:rng[0][1], rng[1][0]:rng[1][1], rng[2][0]:rng[2][1]]
                valid[: rng[0][1] - rng[0][0], : rng[1][1] - rng[1][0], : rng[2][1] - rng[2][0]] = True
                return buf.to(dev), valid
            hk.reset("patch"); hk.patch_layer, hk.region, hk.src = layer, regions[region], src
            lg, pred = infer(pre_b, data_b); cov, dice = measure(pred)
            return cov, dice, float(lg[coreT].mean())
        donor = next(c for c in POOL if c != case)
        for layer in LAYERS:
            for region in ("R1", "R2", "R3"):
                cov, dice, lgc = run_patch(layer, region, "dark")
                out_rows.append({"case": case, "rank": rank, "layer": layer, "region": region, "cond": "patch", "cov": round(cov, 4), "dice": round(dice, 4), "logit_core": round(lgc, 3), "n_windows": len(coords)})
            cov, dice, lgc = run_patch(layer, "R1", donor)
            out_rows.append({"case": case, "rank": rank, "layer": layer, "region": "R1", "cond": "control_unrelated", "cov": round(cov, 4), "dice": round(dice, 4), "logit_core": round(lgc, 3), "n_windows": len(coords)})
        os.remove(tmp)
        print(f"[{ci}/{len(my_cases)}] {case} L{rank}: dark {base['dark'][0]:.2f} (earlier {float(e['cov0']):.2f}) bright {base['bright'][0]:.2f} (earlier {float(e['cov']):.2f}); {time.time() - t0:.0f}s", flush=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    del store
