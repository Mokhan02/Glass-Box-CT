"""W6b training driver (results/w6b_distill/PREREG.md): scripts/w6_train.py (the W6 driver, unchanged in everything not listed here) plus the self-distillation trainer glassbox.seg.w6b_trainer.W6bTrainer. Differences: arms W6b-FULL (BRIGHT edit, whole network) and W6b-DEC (BRIGHT edit, decoder-only); arguments --lam, --k; the trainer class is swapped by config; the report adds the loss parts and the smoke checks; --class-freq N writes how often each distillable class is present in N training patches (teacher masks) and exits. Original W6 driver docstring follows. W6 training driver (results/w6_vista_finetune/PREREG.md): the official NV-Segment-CT continual-learning configuration (`configs/train.json` + `configs/train_continual.json`), with the overlay `scripts/w6_overlay.json` and programmatic arm settings.
Arms: BRIGHT / SHAM / PLAIN (full fine-tune; the augmentation `glassbox.seg.w6_bright.BrightTumourShiftd` is inserted right after the random crop; PLAIN has none), BRIGHT-DEC (BRIGHT edit, encoder frozen: trainable = `image_encoder.up_layers_auto` and `class_head`).
Preflight (asserted before the first iteration): the label set equals [0] + the mapped classes (so no unmapped class can be prompted: PREREG section 4), 1,000 sampled prompt sets contain only those classes, the trainable parameter count of the arm, MONAI / torch versions, the SHA-256 of the weights and of this file. The FINAL model (the last iteration) is saved; no checkpoint is selected on any score.
Usage (cwd anywhere; python of an environment with MONAI and the bundle's requirements):
  python scripts/w6_train.py --bundle <NV-Segment-CT dir> --arm BRIGHT --weights <model.pt> --dataset-dir <Task03 dir> --data-list <datalist.json> --out <dir> [--iterations 8000] [--patch 128] [--mapping 1,1:2,26] [--seed 20261110] [--device cuda] [--smoke] [--workers 4]"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

ARMS = {"W6b-FULL": ("BRIGHT", False), "W6b-DEC": ("BRIGHT", True)}
DEC_TRAINABLE = ("image_encoder.up_layers_auto", "class_head")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_mapping(s):
    return [[int(a), int(b)] for a, b in (x.split(",") for x in s.split(":"))]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--arm", required=True, choices=list(ARMS)); ap.add_argument("--weights", default=None); ap.add_argument("--dataset-dir", required=True); ap.add_argument("--data-list", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--iterations", type=int, default=8000); ap.add_argument("--patch", type=int, default=128); ap.add_argument("--mapping", default="1,1:2,26"); ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--p", type=float, default=0.25); ap.add_argument("--val-cases", type=int, default=10); ap.add_argument("--val-every-epochs", type=int, default=10); ap.add_argument("--lam", type=float, required=True); ap.add_argument("--k", type=int, default=16); ap.add_argument("--class-freq", type=int, default=0); ap.add_argument("--student", default=None)
    a = ap.parse_args(argv); bundle = os.path.abspath(a.bundle); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    sys.path.insert(0, bundle); os.chdir(bundle)      # the bundle's `scripts` package and `bundle_root = .`
    import monai
    import numpy as np
    import torch
    from monai.bundle import ConfigParser
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mapping = parse_mapping(a.mapping); arm_edit, decoder_only = ARMS[a.arm]
    datalist = json.load(open(a.data_list)); n_train = sum(1 for r in datalist["training"] if r["fold"] != 0)
    n_used = min(n_train, 8) if a.smoke else n_train; epochs = max(1, math.ceil(a.iterations / max(n_used, 1))); overlay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w6_overlay.json")
    p = ConfigParser(); p.read_config([os.path.join(bundle, "configs", "train.json"), os.path.join(bundle, "configs", "train_continual.json"), overlay])
    p["dataset_dir"] = a.dataset_dir; p["data_list_file_path"] = os.path.abspath(a.data_list); p["finetune"] = bool(a.weights); p["finetune_model_path"] = a.weights or ""
    p["device"] = f"$torch.device('{a.device}')"; p["patch_size"] = [a.patch] * 3; p["label_mappings"] = {"default": mapping}; p["epochs"] = epochs; p["initialize"] = [f"$monai.utils.set_determinism(seed={a.seed})"]
    p["ckpt_dir"] = os.path.join(out, "ckpt"); p["output_dir"] = os.path.join(out, "eval"); p["n_val_samples"] = a.val_cases; p["val_interval"] = 10 ** 6 if a.smoke else a.val_every_epochs
    p["train#dataloader#num_workers"] = a.workers; p["train#dataloader#persistent_workers"] = a.workers > 0; p["validate#dataloader#num_workers"] = min(a.workers, 2)
    if a.smoke:
        p["n_train_samples"] = n_used      # the smoke test caches only n_used cases; `epochs` above is sized from n_used so that --iterations iterations really run
    # ---- transform order (PREREG W6 amendment 1): the edit runs on HU BEFORE the bundle's intensity clip / rescale (`ScaleIntensityRanged`), which is moved from the cached deterministic chain to right after the crop (and the edit), for ALL arms (PLAIN too), so that every
    # arm sees the same order; the validation preprocessing keeps the official order. Every other intensity augmentation (zoom, low-resolution, smoothing, scale / shift, noise) comes after the rescale.
    det = list(p["train#deterministic_transforms"]); k = next(i for i, t in enumerate(det) if "ScaleIntensityRanged" in t["_target_"]); scale_cfg = det[k]; p["w6_det_full"] = json.loads(json.dumps(det)); del det[k]
    # official behaviour kept: the official chain pads AFTER the rescale, i.e. with 0 in scaled units = a_min in HU; now that the rescale follows, the image is padded with a_min and the label with 0 (two SpatialPadd instead of one)
    kp = next(i for i, t in enumerate(det) if "SpatialPadd" in t["_target_"]); sp = det[kp]
    det[kp:kp + 1] = [{"_target_": "SpatialPadd", "keys": ["image"], "spatial_size": sp["spatial_size"], "mode": "constant", "constant_values": float(scale_cfg["a_min"])}, {"_target_": "SpatialPadd", "keys": ["label"], "spatial_size": sp["spatial_size"], "mode": "constant"}]
    p["train#deterministic_transforms"] = det; p["validate#preprocessing#transforms"] = "$@w6_det_full + [@valid_remap]"
    tl = list(p["train#random_transforms"]); i = next(k2 for k2, t in enumerate(tl) if "RandCropByLabelClassesd" in t["_target_"]); ins = []
    if arm_edit:
        ins.append({"_target_": "glassbox.seg.w6_bright.BrightTumourShiftd", "keys": ["image", "label"], "arm": arm_edit, "p": a.p})
    ins.append(json.loads(json.dumps(scale_cfg))); tl[i + 1:i + 1] = ins; p["train#random_transforms"] = tl
    if decoder_only:
        p["optimizer#params"] = "$[q for q in @network.parameters() if q.requires_grad]"
    label_dict = json.load(open(os.path.join(bundle, "configs", "label_dict.json"))); vals = list(label_dict.values()) if all(isinstance(v, int) for v in label_dict.values()) else list(label_dict.keys())
    dclasses = sorted({int(v) for v in vals} - {0} - {m[1] for m in mapping}); assert len(dclasses) == 123 and max(dclasses) == 132 and 26 not in dclasses and 0 not in dclasses, len(dclasses)      # the guard's 123-class list: every class except background and the tumour class
    p["train#trainer#_target_"] = "glassbox.seg.w6b_trainer.W6bTrainer"; p["train#trainer#w6b"] = {"lam": a.lam, "k": a.k, "classes": dclasses, "seed": a.seed + 1, "weights": a.weights, "checks": bool(a.smoke)}
    p.parse()
    # ---- preflight asserts (PREREG section 4)
    label_set = p.get_parsed_content("label_set"); expect = [0] + [m[1] for m in mapping]; assert list(label_set) == expect, (label_set, expect)
    from monai.apps.vista3d.sampler import sample_prompt_pairs
    seen = set(); rs = random.Random(0); lab = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    for _ in range(1000):
        lab[:] = 0
        for g in (m[1] for m in mapping):
            if rs.random() < 0.5:
                lab[0, 0, rs.randrange(16), rs.randrange(16), rs.randrange(16)] = g
        lp, pt, pl, pc = sample_prompt_pairs(lab, label_set, max_point=5, include_background=False, drop_label_prob=0.0, drop_point_prob=1.0)
        if pc is not None:
            seen |= set(int(v) for v in pc.flatten().tolist())
    assert seen <= set(expect), f"prompted classes outside the label set: {sorted(seen - set(expect))}"
    names_tr = [type(t).__name__ for t in p.get_parsed_content("train#preprocessing").transforms]; names_va = [type(t).__name__ for t in p.get_parsed_content("validate#preprocessing").transforms]
    ic = names_tr.index("RandCropByLabelClassesd"); isc = names_tr.index("ScaleIntensityRanged")
    assert names_tr.count("ScaleIntensityRanged") == 1 and isc == ic + (2 if arm_edit else 1), names_tr      # crop, [edit], rescale, in this order, and nothing else between
    if arm_edit:
        assert names_tr[ic + 1] == "BrightTumourShiftd" and names_tr.index("BrightTumourShiftd") < isc, names_tr
    later = [x for x in ("RandZoomd", "RandSimulateLowResolutiond", "RandGaussianSmoothd", "RandScaleIntensityd", "RandShiftIntensityd", "RandGaussianNoised") if x in names_tr]; assert later and all(names_tr.index(x) > isc for x in later), names_tr      # every other intensity augmentation after the rescale
    assert names_tr.index("Orientationd") < ic and names_tr.index("Spacingd") < ic, names_tr      # the edit sees the resampled (1.5 mm) HU patch
    assert names_va.index("ScaleIntensityRanged") < names_va.index("Orientationd") < names_va.index("Spacingd") and names_va[-1] == "Relabeld", names_va      # validation: official order
    print("[w6b] train transform order:", names_tr, flush=True); print("[w6b] validation transform order:", names_va, flush=True)
    network = p.get_parsed_content("network")
    if decoder_only:
        for n, q in network.named_parameters():
            q.requires_grad = n.startswith(DEC_TRAINABLE)
    n_tr = sum(q.numel() for q in network.parameters() if q.requires_grad); n_all = sum(q.numel() for q in network.parameters())
    if a.class_freq:
        return class_freq(p, network, a, out, dclasses, torch, np)
    trainer = p.get_parsed_content("train#trainer")
    from ignite.engine import Events
    stats = {"loss": [], "sec": [], "edited": [], "t0": time.time()}
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    @trainer.on(Events.ITERATION_COMPLETED)
    def _log(e):
        o = e.state.output; o = o[0] if isinstance(o, (list, tuple)) else o
        if isinstance(o, dict) and "loss" in o:
            stats["loss"].append(float(o["loss"]))
        stats["sec"].append(time.time()); b = e.state.batch; b = b if isinstance(b, (list, tuple)) else [b]
        for it in b:
            if isinstance(it, dict) and "bright_applied" in it:
                v = it["bright_applied"]; stats["edited"].append(int(v.sum()) if hasattr(v, "sum") else int(v))
        if e.state.iteration % 10 == 0:
            print(f"[w6b {a.arm}] it {e.state.iteration}/{a.iterations} loss {stats['loss'][-1] if stats['loss'] else float('nan'):.4f}", flush=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(once=a.iterations), lambda e: e.terminate())
    trainer.add_event_handler(Events.ITERATION_COMPLETED, __import__("ignite").handlers.TerminateOnNan())
    trainer.run()
    trainer.w6b_checks["teacher_param_abs_sum_end"] = float(sum(q.detach().abs().sum().item() for q in trainer.teacher.parameters())); done = trainer.state.iteration; torch.save(network.state_dict(), os.path.join(out, "final_model.pt"))
    dt = np.diff(stats["sec"]) if len(stats["sec"]) > 1 else np.array([float("nan")])
    rep = {"arm": a.arm, "seed": a.seed, "iterations_done": int(done), "iterations_requested": a.iterations, "patch": a.patch, "mapping": mapping, "label_set": list(label_set), "transform_order_train": names_tr, "transform_order_validation": names_va, "trainable_params": int(n_tr), "all_params": int(n_all),
           "final_model_sha256": sha256(os.path.join(out, "final_model.pt")), "weights_sha256": sha256(a.weights) if a.weights else None, "driver_sha256": sha256(os.path.abspath(__file__)), "overlay_sha256": sha256(overlay), "monai": monai.__version__, "torch": torch.__version__,
           "seconds_per_iteration_median": float(np.nanmedian(dt)), "loss_first": stats["loss"][:5], "loss_last": stats["loss"][-5:], "loss_all_finite": bool(np.all(np.isfinite(stats["loss"]))) if stats["loss"] else None,
           "edited_patch_fraction": (float(np.mean(stats["edited"])) if stats["edited"] else None), "n_batches_seen": len(stats["edited"]), "wall_seconds": time.time() - stats["t0"],
           "lam": a.lam, "k": a.k, "distill_classes_n": len(dclasses), "loss_task_mean": float(np.mean(trainer.w6b_hist["task"])), "loss_distill_bce_mean": float(np.mean(trainer.w6b_hist["distill"])), "loss_distill_kl_mean": float(np.mean(trainer.w6b_hist["kl"])), "loss_distill_kl_last100_mean": float(np.mean(trainer.w6b_hist["kl"][-100:])),
           "loss_task_first5": trainer.w6b_hist["task"][:5], "loss_distill_kl_first5": trainer.w6b_hist["kl"][:5], "checks": trainer.w6b_checks, "trainer_sha256": sha256(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glassbox", "seg", "w6b_trainer.py")), "loss_sha256": sha256(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glassbox", "seg", "w6b_loss.py"))}
    if a.device.startswith("cuda"):
        rep["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9; rep["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    json.dump(rep, open(os.path.join(out, "run_report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
    return rep


def class_freq(p, network, a, out, dclasses, torch, np):
    """--class-freq N: for N training patches (the arm's own training pipeline, same edit), the frozen baseline's masks for every distillable class: the share of patches in which a class has at least 1,000 (and at least 100) voxels at p > 0.5. Training data only; no evaluation image is read."""
    import copy
    teacher = copy.deepcopy(network); teacher.load_state_dict(torch.load(a.weights, map_location="cpu")); teacher.to(a.device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = None
    if a.student:      # optional: a trained model (e.g. BRIGHT) whose divergence from the teacher is measured on the same patches = the scale the distillation term would have for a student that drifted like it did
        student = copy.deepcopy(network); student.load_state_dict(torch.load(a.student, map_location="cpu")); student.to(a.device).eval()
    from glassbox.seg.w6b_loss import distill_bce
    kls = np.zeros(len(dclasses)); dl = p.get_parsed_content("train#dataloader"); n = 0; cnt1000 = np.zeros(len(dclasses)); cnt100 = np.zeros(len(dclasses)); vox = np.zeros(len(dclasses)); cv_all = torch.tensor(dclasses, device=a.device).long().unsqueeze(-1)
    while n < a.class_freq:
        for batch in dl:
            items = batch if isinstance(batch, (list, tuple)) else [batch]
            for it in items:
                x = it["image"]; x = x.unsqueeze(0) if x.ndim == 4 else x
                x = x.to(a.device)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    for j in range(0, len(dclasses), 41):
                        lo = teacher(input_images=x, point_coords=None, point_labels=None, class_vector=cv_all[j:j + 41])
                        c = (lo.float() > 0).flatten(2).sum(-1).squeeze(1).cpu().numpy() if lo.ndim == 5 else (lo.float() > 0).flatten(1).sum(-1).cpu().numpy()      # logit > 0 is p > 0.5
                        cnt1000[j:j + 41] += c >= 1000; cnt100[j:j + 41] += c >= 100; vox[j:j + 41] += c
                        if student is not None:
                            ls = student(input_images=x, point_coords=None, point_labels=None, class_vector=cv_all[j:j + 41])
                            for q in range(lo.shape[0]):
                                kls[j + q] += float(distill_bce(ls[q:q + 1], lo[q:q + 1])[1])
                n += 1
                if n % 25 == 0:
                    print(f"[w6b class-freq] {n}/{a.class_freq} patches", flush=True)
                if n >= a.class_freq:
                    break
            if n >= a.class_freq:
                break
    names = {int(v): k for k, v in json.load(open(os.path.join(os.getcwd(), "configs", "label_dict.json"))).items()} if True else {}
    rep = {"n_patches": n, "patch": a.patch, "threshold": "logit > 0 (p > 0.5)", "classes": {str(c): {"name": names.get(c), "share_ge_1000_voxels": float(cnt1000[i] / n), "share_ge_100_voxels": float(cnt100[i] / n), "mean_voxels": float(vox[i] / n), **({"student_kl_mean": float(kls[i] / n)} if student is not None else {})} for i, c in enumerate(dclasses)}}
    if student is not None:
        rep["student"] = a.student; rep["student_kl_mean_over_classes"] = float(kls.mean() / n)
    json.dump(rep, open(os.path.join(out, "class_freq.json"), "w"), indent=1); print(json.dumps({k: rep["classes"][k] for k in ("1", "3", "5", "14") if k in rep["classes"]}, indent=1)); return rep


if __name__ == "__main__":
    main()
