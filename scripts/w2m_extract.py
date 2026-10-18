"""W2-M M1 step 1 (results/w2_mech/PREREG.md): mid-encoder tokens on the C4KC-KiTS main-set cases, mirroring scripts/voxel_probe_extract.py.
Per case: 12 mm mid tokens (192-d, `HOOKS["mid"]`) inside the GT kidney+mass dilated 10 mm, each with: TUMOUR (a main-set primary lesion covers >= 0.5 of the cell, no other Mass in the cell), KID (no Mass in the cell,
Kidney coverage >= 0.5), the lesion rank and bright/dark sign, interior flag (> 2 mm from the nearest known non-Mass voxel) and 14 local raw-HU features. Pilot lesions and non-primary Mass components are dropped.
--skip-windows runs only the windows that intersect the ROI (token identity with the full grid is checked by --identity-check before it may be used).
Usage: python scripts/w2m_extract.py --model trained|random --out DIR [--cases ...] [--limit N] [--skip-windows] [--identity-check N]"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from monai.data import MetaTensor  # noqa: E402
from monai.inferers import SlidingWindowInfererAdapt  # noqa: E402
from monai.transforms import LoadImage  # noqa: E402
from scipy import ndimage  # noqa: E402

from glassbox.audit.contrast import select_lesions  # noqa: E402
from glassbox.data.preprocess import HU_MAX, HU_MIN, PATCH_SIZE, SW_OVERLAP, build_pair_transforms  # noqa: E402
from glassbox.extract.hooks import HOOKS, RoiTokenRecorder  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="trained", choices=["trained", "random"]); ap.add_argument("--out", required=True)
ap.add_argument("--root", default="/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"); ap.add_argument("--weights", default="/lambda/nfs/CTGate/glassbox/cache/weights/model.pt")
ap.add_argument("--cases", nargs="*"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--skip-windows", action="store_true"); ap.add_argument("--identity-check", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda"); VOX = 1.5; STRIDE = HOOKS["mid"][0]
if a.model == "trained":
    net = load_network(a.weights, dev)
else:
    from monai.networks.nets import vista3d132
    torch.manual_seed(0); net = vista3d132(encoder_embed_dim=48, in_channels=1).to(dev).eval()
load = LoadImage(image_only=True)
pilot = {(r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv")) if r["organ"] == "kidney"}
table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2}
main_cases = sorted({c for (c, r) in table if (c, r) not in pilot})
cases = (a.cases or main_cases)[: a.limit or None]


class SkipNet:
    """Predictor wrapper: windows with no ROI voxel are not run (they would record no token); a zero template stands in for their output."""
    def __init__(self, net, rec):
        self.net, self.rec, self.template, self.ran, self.skipped = net, rec, None, 0, 0

    def __call__(self, *args, **kwargs):
        sl = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]; lab = self.rec.label[tuple(slice(s.start, s.stop) for s in sl[-3:])]
        if self.template is not None and not bool((lab == 2).any()):
            self.skipped += 1; return torch.zeros_like(self.template)
        out = self.net(*args, **kwargs); self.template = out.detach(); self.ran += 1; return out

    def __getattr__(self, name):
        return getattr(self.net, name)


def box_sums(II, lo, hi):
    x0, y0, z0 = lo.T; x1, y1, z1 = hi.T
    return (II[x1, y1, z1] - II[x0, y1, z1] - II[x1, y0, z1] - II[x1, y1, z0] + II[x0, y0, z1] + II[x0, y1, z0] + II[x1, y0, z0] - II[x0, y0, z0])


def integral(v):
    II = np.zeros([s + 1 for s in v.shape], np.float64); II[1:, 1:, 1:] = v.astype(np.float64).cumsum(0).cumsum(1).cumsum(2); return II


def tokens_for(case, skip):
    d = os.path.join(a.root, case); img, seg = load(os.path.join(d, "ct.nii.gz")), load(os.path.join(d, "seg.nii.gz")); segn = seg.numpy()
    import nibabel as nib
    sp = tuple(float(z) for z in nib.load(os.path.join(d, "ct.nii.gz")).header.get_zooms()[:3])
    ann = (segn > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann)
    lab_c, sel = select_lesions((segn == 2)[:, :, ann], sp)
    lab = np.zeros(segn.shape, np.int16); lab[segn == 1] = 1; lab[segn == 2] = 4; lab[:, :, ~ann] = 3             # 4 = Mass not in a main-set primary lesion (pilot lesions, other components)
    info = []
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        ref = table[(case, rank)]; assert abs(float(ref["diam_mm"]) - diam) < 0.01, f"lesion mismatch {case} {rank}"
        if (case, rank) in pilot:
            continue
        bright = float(ref["contrast_hu"]) > 0; code = (20 if bright else 30) + rank
        full = np.zeros(segn.shape, bool); full[:, :, ann] = lab_c == cid; lab[full] = code; info.append({"rank": rank, "bright": bright, "code": code, "diam_mm": diam})
    pair = build_pair_transforms()({"image": img, "label": MetaTensor(torch.from_numpy(lab), meta=img.meta.copy(), affine=img.affine)})
    x = pair["image"].unsqueeze(0).to(dev); L = pair["label"][0].as_tensor().numpy().astype(np.int16); H, W, D = L.shape
    hu = x[0, 0].cpu().numpy() * (HU_MAX - HU_MIN) + HU_MIN
    kid, known = L == 1, L != 3; les = L >= 20; mass = les | (L == 4)
    roi = (ndimage.distance_transform_edt(~(kid | mass)) * VOX) <= 10.0
    rec = RoiTokenRecorder(net, torch.from_numpy(roi), seed=0); rec.label = rec.label.to(dev)
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    pred_net = SkipNet(net, rec) if skip else net
    try:
        with torch.no_grad():
            sw(x, pred_net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[5]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
        tk = rec.collect()
    finally:
        rec.close()
    II = {k: integral(v) for k, v in (("kid", kid), ("known", known), ("mass", mass), ("one", np.ones_like(kid)))}
    for inf in info:
        II[inf["code"]] = integral(L == inf["code"])
    Ihu1, Ihu2 = integral(hu), integral(hu.astype(np.float64) ** 2)
    edt = ndimage.distance_transform_edt(mass | ~known) * VOX
    xyz = tk["mid_xyz"].astype(np.int64); xs = tk["mid_x"]; c = np.clip(xyz, 0, np.array([H, W, D]) - 1)
    lo = np.clip(xyz - STRIDE // 2, 0, [H, W, D]); hi = np.clip(lo + STRIDE, 0, [H, W, D])
    n_cell = box_sums(II["one"], lo, hi); n_known = box_sums(II["known"], lo, hi); full_ = n_cell >= 0.5 * STRIDE ** 3; kf = n_known / np.maximum(n_cell, 1)
    s_kid, s_mass = box_sums(II["kid"], lo, hi), box_sums(II["mass"], lo, hi)
    cov = {inf["code"]: box_sums(II[inf["code"]], lo, hi) / np.maximum(n_known, 1) for inf in info}
    best = np.zeros(len(xyz), np.int16); bestcov = np.zeros(len(xyz))
    for inf in info:
        m = cov[inf["code"]] > bestcov; best[m] = inf["code"]; bestcov[m] = cov[inf["code"]][m]
    s_les_best = np.zeros(len(xyz))
    for inf in info:
        s_les_best = np.where(best == inf["code"], box_sums(II[inf["code"]], lo, hi), s_les_best)
    other_mass = (s_mass - s_les_best) > 0                                               # Mass in the cell that is not the assigned main-set lesion
    is_t = full_ & (kf >= 0.5) & (bestcov >= 0.5) & ~other_mass & (best > 0)
    is_k = full_ & (kf >= 0.5) & (s_mass == 0) & (s_kid / np.maximum(n_known, 1) >= 0.5)
    selx = np.flatnonzero(is_t | is_k); hf = np.zeros((len(selx), 14), np.float32)
    for scale, off in ((1, 0), (3, 7)):
        half = STRIDE * scale // 2; lo2 = np.clip(xyz[selx] - half, 0, [H, W, D]); hi2 = np.clip(xyz[selx] + (STRIDE * scale - half), 0, [H, W, D])
        n2 = np.maximum(box_sums(II["one"], lo2, hi2), 1); m_ = box_sums(Ihu1, lo2, hi2) / n2; v_ = box_sums(Ihu2, lo2, hi2) / n2 - m_ ** 2
        hf[:, off] = m_; hf[:, off + 1] = np.sqrt(np.maximum(v_, 0))
        for k, (l_, h_) in enumerate(zip(lo2, hi2)):
            hf[k, off + 2:off + 7] = np.percentile(hu[l_[0]:h_[0], l_[1]:h_[1], l_[2]:h_[2]], [5, 25, 50, 75, 95])
    bright_of = {inf["code"]: inf["bright"] for inf in info}
    lesion_with_token = {inf["rank"]: bool(((best == inf["code"]) & is_t).any()) for inf in info}
    out = {"mid_x": xs[selx], "mid_xyz": xyz[selx].astype(np.int16), "mid_is_tum": is_t[selx], "mid_is_kid": is_k[selx], "mid_lesion_code": best[selx], "mid_bright": np.array([bright_of.get(int(b), False) for b in best[selx]]),
           "mid_interior": (edt[c[selx, 0], c[selx, 1], c[selx, 2]] > 2.0), "mid_hu14": hf.astype(np.float16), "mid_patch": tk["mid_patch"][selx],
           "lesion_ranks": np.array([i["rank"] for i in info]), "lesion_bright": np.array([i["bright"] for i in info]), "lesion_has_token": np.array([lesion_with_token[i["rank"]] for i in info]), "lesion_diam_mm": np.array([i["diam_mm"] for i in info])}
    stats = {"ran": getattr(pred_net, "ran", None), "skipped": getattr(pred_net, "skipped", None)}
    return out, stats


def canon(o):
    """Order-independent token identity: rows sorted by (xyz, feature bytes); the recorder's window index is dropped."""
    key = np.lexsort(np.concatenate([o["mid_x"].astype(np.float32).T[::-1], o["mid_xyz"].astype(np.int32).T[::-1]]))
    return {k: o[k][key] for k in ("mid_x", "mid_xyz", "mid_is_tum", "mid_is_kid", "mid_lesion_code", "mid_hu14")}


os.makedirs(a.out, exist_ok=True)
if a.identity_check:
    ok = True
    for case in cases[: a.identity_check]:
        t0 = time.time(); full, _ = tokens_for(case, False); t_full = time.time() - t0; t0 = time.time(); fast, st = tokens_for(case, True); t_fast = time.time() - t0
        A, Bq = canon(full), canon(fast); same = all(A[k].shape == Bq[k].shape and np.array_equal(A[k], Bq[k]) for k in A)
        print(f"identity {case}: tokens {len(A['mid_xyz'])} vs {len(Bq['mid_xyz'])}; identical {same}; windows run {st['ran']} skipped {st['skipped']}; full {t_full:.0f}s fast {t_fast:.0f}s", flush=True); ok &= same
    print("IDENTITY CHECK", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
for ci, case in enumerate(cases, 1):
    dst = os.path.join(a.out, f"{case}.npz")
    if os.path.exists(dst):
        continue
    t0 = time.time(); out, st = tokens_for(case, a.skip_windows); np.savez_compressed(dst + ".part.npz", **out); os.replace(dst + ".part.npz", dst)
    print(f"[{ci}/{len(cases)}] {case} tokens tum/kid {int(out['mid_is_tum'].sum())}/{int(out['mid_is_kid'].sum())}; lesions {len(out['lesion_ranks'])} with token {int(out['lesion_has_token'].sum())}; {time.time() - t0:.0f}s", flush=True)
