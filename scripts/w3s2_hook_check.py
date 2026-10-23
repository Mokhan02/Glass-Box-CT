"""W3 Stage 2, first GPU session: real-case hook check for models A and D (results/w3_models/PREREG_stage2.md). STOP (non-zero exit) if any check fails.
Checks per model, on real HCC cases with the trained checkpoints:
  1. SHAPES: the hooked tensors have the pre-registered channels / strides on a window centred on a real lesion (A: swinViT.layers2[0] output (1,192,12,12,12), hidden states 2 and 4, backbone.decoder1 (1,48,96,96,96); D: encoder stages 3/4/5 and decoder.stages[-1]; A U-Net: down_tr256[0], down_tr512[0], up_tr64).
  2. PROPAGATION: a channel-varying edit (seeded noise, 0.5 x std; a constant offset is invisible to LayerNorm / InstanceNorm) applied at the intervention tensor (A: layers2[0] output; D: encoder.stages[3] output) reaches BOTH consumers: the skip / hidden-state copy equals the edited tensor (A: hidden state 2 = LayerNorm(edited layers2[0] output); D: the encoder's returned skip 3 = edited stage 3 output exactly)
     AND the deeper path changes (A: hidden states 3 and 4; D: skip 4 and the first decoder stage), and the final logits change.
  3. TRAINED-CHECKPOINT SANITY: the unhooked pipeline reproduces Stage 1 on 3 fixed cases (the first 3 of the sorted cohort): lesion coverage within 1e-3 and identical detection flags (GPU non-determinism was measured at 3e-4).
Model A (monai 0.9.0, numpy 1.26: venv_a) and model D (nnunetv2: venv_nnunet) need different interpreters, so the check runs once per model and the two result files are both required:
Usage (cwd = repo checkout, GPU): python scripts/w3s2_hook_check.py A <outdir> <repo_A> <swin weights> <unet weights>      (venv_a)
                                  python scripts/w3s2_hook_check.py D <outdir> <nnU-Net model folder>                      (venv_nnunet)"""
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

RUN = sys.argv[1]; assert RUN in ("A", "D"); outdir = sys.argv[2]
repo_a, swin_w, unet_w = (sys.argv[3:6] if RUN == "A" else (None, None, None)); mf = sys.argv[3] if RUN == "D" else None; os.makedirs(outdir, exist_ok=True)
if repo_a:
    sys.path.insert(0, repo_a)
dev = torch.device("cuda"); R = {"checks": {}}; CASE = "HCC_004_414785"
root = L.case_root("hcc", CASE, L.HCC_ROOT); ct = os.path.join(root, CASE, "ct.nii.gz"); seg = np.asarray(nib.load(os.path.join(root, CASE, "seg.nii.gz")).dataobj)
ok_all = True


def perturb(out):
    """Channel-varying edit (fixed seed, 0.5 x the tensor's std). A constant offset would be a bad test: LayerNorm (A) and InstanceNorm after a convolution (D) are invariant to it."""
    g = torch.Generator(device=out.device); g.manual_seed(0); return out + 0.5 * out.std() * torch.randn(out.shape, device=out.device, generator=g)


def rec(name, ok, detail):
    global ok_all
    ok_all &= bool(ok); R["checks"][name] = {"pass": bool(ok), **detail}; print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, default=float)[:400], flush=True)


idx = np.argwhere(seg == 2); frac = idx.mean(0) / np.array(seg.shape)
if RUN == "A":
    # ---------------- model A (and U-Net shapes) ----------------
    from monai.transforms import AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord  # noqa: E402
    from model.Universal_model import Universal_model  # noqa: E402
    from utils.utils import NUM_CLASS  # noqa: E402

    tf = Compose([LoadImaged(keys=["image"]), AddChanneld(keys=["image"]), Orientationd(keys=["image"], axcodes="RAS"), Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
                  ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=["image"], source_key="image"), ToTensord(keys=["image"])])
    x = tf({"image": ct})["image"].unsqueeze(0).to(dev)
    cen = [int(round(f * n)) for f, n in zip(frac, x.shape[2:])]; o = [max(0, min(c - 48, n - 96)) for c, n in zip(cen, x.shape[2:])]
    win = x[:, :, o[0]:o[0] + 96, o[1]:o[1] + 96, o[2]:o[2] + 96]


    def load_universal(backbone, wts):
        m = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone=backbone, encoding="word_embedding"); store = m.state_dict()
        for k, v in torch.load(wts, map_location="cpu")["net"].items():
            store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
        m.load_state_dict(store); return m.to(dev).eval()


    mA = load_universal("swinunetr", swin_w); caps = {}; mode = {"offset": None}
    sw = mA.backbone.swinViT
    sw.layers2[0].register_forward_hook(lambda m, a, out: (caps.__setitem__("x2", out.detach().clone()) if mode["offset"] is None else None) or (perturb(out) if mode["offset"] is not None else None))
    sw.register_forward_hook(lambda m, a, out: caps.__setitem__("hs", [t.detach().clone() for t in out]))
    mA.backbone.decoder1.register_forward_hook(lambda m, a, out: caps.__setitem__("dec", out.detach().clone()))
    with torch.no_grad():
        base_logits = mA(win)
    base = {k: caps[k] for k in ("x2", "hs", "dec")}
    rec("A shapes", tuple(base["x2"].shape) == (1, 192, 12, 12, 12) and tuple(base["hs"][2].shape) == (1, 192, 12, 12, 12) and tuple(base["hs"][4].shape) == (1, 768, 3, 3, 3) and tuple(base["dec"].shape) == (1, 48, 96, 96, 96),
        {"layers2[0]": list(base["x2"].shape), "hs2": list(base["hs"][2].shape), "hs4": list(base["hs"][4].shape), "decoder1": list(base["dec"].shape)})
    ln = F.layer_norm(base["x2"].permute(0, 2, 3, 4, 1), [192]).permute(0, 4, 1, 2, 3)
    rec("A hidden-state-2 = LayerNorm(layers2[0] output)", torch.allclose(ln, base["hs"][2], atol=1e-4), {"max_abs_diff": float((ln - base["hs"][2]).abs().max())})
    mode["offset"] = True
    with torch.no_grad():
        ed_logits = mA(win)
    hs_ed = caps["hs"]; ln_ed = F.layer_norm(perturb(base["x2"]).permute(0, 2, 3, 4, 1), [192]).permute(0, 4, 1, 2, 3)
    skip_ok = torch.allclose(hs_ed[2], ln_ed, atol=1e-4); deeper = [float((hs_ed[i] - base["hs"][i]).abs().max()) for i in (3, 4)]; logit_change = float((ed_logits - base_logits).abs().max())
    rec("A propagation (skip copy AND deeper path)", skip_ok and min(deeper) > 1e-2 and logit_change > 1e-2, {"hs2_equals_LN(edited)": bool(skip_ok), "hs3_hs4_max_abs_change": deeper, "logit_max_abs_change": logit_change})
    mode["offset"] = None; del mA
    mU = load_universal("unet", unet_w); cu = {}
    mU.backbone.down_tr256.register_forward_hook(lambda m, a, out: cu.__setitem__("d256", tuple(out[0].shape))); mU.backbone.down_tr512.register_forward_hook(lambda m, a, out: cu.__setitem__("d512", tuple(out[0].shape)))
    mU.backbone.up_tr64.register_forward_hook(lambda m, a, out: cu.__setitem__("u64", tuple(out.shape)))
    with torch.no_grad():
        mU(win)
    rec("A U-Net shapes", cu == {"d256": (1, 256, 12, 12, 12), "d512": (1, 512, 12, 12, 12), "u64": (1, 64, 96, 96, 96)}, {k: list(v) for k, v in cu.items()}); del mU

if RUN == "D":
    # ---------------- model D ----------------
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True, device=dev, verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(mf, use_folds=(0,), checkpoint_name="checkpoint_final.pth"); net = pred.network.to(dev).eval(); cm = pred.configuration_manager; ps = list(cm.patch_size)
    data, _, _ = cm.preprocessor_class(verbose=False).run_case([ct], None, pred.plans_manager, cm, pred.dataset_json)
    z = [int(round(f * n)) for f, n in zip(frac[::-1], data.shape[1:])]; lo = [max(0, min(zc - p // 2, n - p)) for zc, p, n in zip(z, ps, data.shape[1:])]
    patch = torch.from_numpy(np.ascontiguousarray(data[:, lo[0]:lo[0] + ps[0], lo[1]:lo[1] + ps[1], lo[2]:lo[2] + ps[2]])).float()[None].to(dev)
    cd = {}; dmode = {"offset": None}
    net.encoder.stages[3].register_forward_hook(lambda m, a, out: perturb(out) if dmode["offset"] is not None else None)
    for i in (3, 4, 5):
        net.encoder.stages[i].register_forward_hook(lambda m, a, out, i=i: cd.__setitem__(f"s{i}", tuple(out.shape)))
    net.decoder.stages[-1].register_forward_hook(lambda m, a, out: cd.__setitem__("dec", tuple(out.shape))); net.decoder.stages[0].register_forward_hook(lambda m, a, out: cd.__setitem__("dec0", out.detach().clone()))
    net.encoder.register_forward_hook(lambda m, a, out: cd.__setitem__("skips", [t.detach().clone() for t in out]))
    with torch.no_grad():
        y0 = net(patch)
    y0 = y0[0] if isinstance(y0, (list, tuple)) else y0; b_skips, b_dec0 = cd["skips"], cd["dec0"]
    rec("D shapes", cd["s3"] == (1, 256, 16, 16, 16) and cd["s4"] == (1, 320, 8, 8, 8) and cd["s5"] == (1, 320, 4, 4, 4) and cd["dec"] == (1, 32, 128, 128, 128), {k: list(v) for k, v in cd.items() if isinstance(v, tuple)})
    dmode["offset"] = True
    with torch.no_grad():
        y1 = net(patch)
    y1 = y1[0] if isinstance(y1, (list, tuple)) else y1; e_skips = cd["skips"]
    skip3_ok = torch.allclose(e_skips[3], perturb(b_skips[3]), atol=1e-5); deeper = [float((e_skips[i] - b_skips[i]).abs().max()) for i in (4, 5)]; d0 = float((cd["dec0"] - b_dec0).abs().max()); lc = float((y1 - y0).abs().max())
    rec("D propagation (skip 3 AND deeper path)", skip3_ok and min(deeper) > 1e-2 and d0 > 1e-2 and lc > 1e-2, {"skip3_equals_edited_stage3": bool(skip3_ok), "skip4_skip5_max_abs_change": deeper, "first_decoder_stage_change": d0, "logit_max_abs_change": lc})
    dmode["offset"] = None

# ---------------- trained-checkpoint sanity: unhooked output reproduces Stage 1 on 3 cases ----------------
table = L.lesion_table("hcc"); cases = sorted({k[0] for k in table})[:3]; stage1 = "results/w3_models/stage1"
import csv  # noqa: E402

for fam, csvp, runner in (("A", f"{stage1}/A_swin_hcc/w3_lesions_A_swin_hcc.csv", None), ("D", f"{stage1}/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv", None)):
    if fam != RUN:
        continue
    ref = {(r["arm"], r["case"], r["rank"]): r for r in csv.DictReader(open(csvp))}; rows = []
    if fam == "A":
        sub = os.path.join(outdir, "sanity_A"); subprocess.run([sys.executable, "scripts/w3_modelA_run.py", repo_a, swin_w, "swin", "hcc", sub, "--limit", "3"], check=True, env={**os.environ, "PYTHONPATH": "."}, capture_output=True)
        rows = [r for r in csv.DictReader(open(os.path.join(sub, "w3_lesions_A_swin_hcc.csv"))) if r["arm"] == "A_swin_shipped"]; key = lambda r: ("A_swin_shipped", r["case"], r["rank"])      # noqa: E731
    else:
        ind, outp = os.path.join(outdir, "sanity_D_in"), os.path.join(outdir, "sanity_D_out"); os.makedirs(ind, exist_ok=True); os.makedirs(outp, exist_ok=True)
        lists = []
        for c in cases:
            dst = os.path.join(ind, f"{c}_0000.nii.gz")
            if not os.path.lexists(dst):
                os.symlink(os.path.join(L.HCC_ROOT, c, "ct.nii.gz"), dst)
            lists.append([dst])
        subprocess.run(["nnUNetv2_predict", "-i", ind, "-o", outp, "-d", "3", "-c", "3d_fullres", "-tr", "nnUNetTrainerW3Seed", "-f", "0", "-chk", "checkpoint_final.pth"], check=True, capture_output=True)      # the Stage 1 command (a library call would spawn workers that re-import this script)
        for c in cases:
            lab = np.asarray(nib.load(os.path.join(outp, f"{c}.nii.gz")).dataobj); r_, _ = L.lesion_rows("hcc", c, L.case_root("hcc", c, L.HCC_ROOT), table, {"D_nnunet": lab == 2}); rows += [r for r in r_]
        key = lambda r: ("D_nnunet", r["case"], str(r["rank"]))      # noqa: E731
    diffs = [(abs(float(ref[key(r)]["coverage"]) - float(r["coverage"])), (float(ref[key(r)]["coverage"]) >= 0.1) == (float(r["coverage"]) >= 0.1)) for r in rows]
    rec(f"{fam} unhooked output reproduces Stage 1 on 3 cases", bool(diffs) and max(d for d, _ in diffs) <= 1e-3 and all(f for _, f in diffs), {"cases": cases, "n_lesions": len(diffs), "max_abs_coverage_diff": max((d for d, _ in diffs), default=None)})
R["all_pass"] = bool(ok_all); json.dump(R, open(os.path.join(outdir, f"hook_check_session1_{RUN}.json"), "w"), indent=1, default=float); print("ALL PASS" if ok_all else "FAILED: STOP", flush=True); sys.exit(0 if ok_all else 1)
