"""W6 training driver (results/w6_vista_finetune/PREREG.md): the official NV-Segment-CT continual-learning configuration (`configs/train.json` + `configs/train_continual.json`), with the overlay `scripts/w6_overlay.json` and programmatic arm settings.
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

ARMS = {"BRIGHT": ("BRIGHT", False), "SHAM": ("SHAM", False), "PLAIN": (None, False), "BRIGHT-DEC": ("BRIGHT", True)}
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
    ap.add_argument("--device", default="cuda"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--p", type=float, default=0.25); ap.add_argument("--val-cases", type=int, default=10); ap.add_argument("--val-every-epochs", type=int, default=10)
    a = ap.parse_args(argv); bundle = os.path.abspath(a.bundle); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    sys.path.insert(0, bundle); os.chdir(bundle)      # the bundle's `scripts` package and `bundle_root = .`
    import monai
    import numpy as np
    import torch
    from monai.bundle import ConfigParser
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mapping = parse_mapping(a.mapping); arm_edit, decoder_only = ARMS[a.arm]
    datalist = json.load(open(a.data_list)); n_train = sum(1 for r in datalist["training"] if r["fold"] != 0)
    epochs = max(1, math.ceil(a.iterations / max(n_train, 1))); overlay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w6_overlay.json")
    p = ConfigParser(); p.read_config([os.path.join(bundle, "configs", "train.json"), os.path.join(bundle, "configs", "train_continual.json"), overlay])
    p["dataset_dir"] = a.dataset_dir; p["data_list_file_path"] = os.path.abspath(a.data_list); p["finetune"] = bool(a.weights); p["finetune_model_path"] = a.weights or ""
    p["device"] = f"$torch.device('{a.device}')"; p["patch_size"] = [a.patch] * 3; p["label_mappings"] = {"default": mapping}; p["epochs"] = epochs; p["initialize"] = [f"$monai.utils.set_determinism(seed={a.seed})"]
    p["ckpt_dir"] = os.path.join(out, "ckpt"); p["output_dir"] = os.path.join(out, "eval"); p["n_val_samples"] = a.val_cases; p["val_interval"] = 10 ** 6 if a.smoke else a.val_every_epochs
    p["train#dataloader#num_workers"] = a.workers; p["train#dataloader#persistent_workers"] = a.workers > 0; p["validate#dataloader#num_workers"] = min(a.workers, 2)
    if a.smoke:
        p["n_train_samples"] = min(n_train, 8)
    if decoder_only:
        p["optimizer#params"] = "$[q for q in @network.parameters() if q.requires_grad]"
    if arm_edit:
        tl = list(p["train#random_transforms"]); i = next(k for k, t in enumerate(tl) if "RandCropByLabelClassesd" in t["_target_"])
        tl.insert(i + 1, {"_target_": "glassbox.seg.w6_bright.BrightTumourShiftd", "keys": ["image", "label"], "arm": arm_edit, "p": a.p}); p["train#random_transforms"] = tl
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
    network = p.get_parsed_content("network")
    if decoder_only:
        for n, q in network.named_parameters():
            q.requires_grad = n.startswith(DEC_TRAINABLE)
    n_tr = sum(q.numel() for q in network.parameters() if q.requires_grad); n_all = sum(q.numel() for q in network.parameters())
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
            print(f"[w6 {a.arm}] it {e.state.iteration}/{a.iterations} loss {stats['loss'][-1] if stats['loss'] else float('nan'):.4f}", flush=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(once=a.iterations), lambda e: e.terminate())
    trainer.add_event_handler(Events.ITERATION_COMPLETED, __import__("ignite").handlers.TerminateOnNan())
    trainer.run()
    done = trainer.state.iteration; torch.save(network.state_dict(), os.path.join(out, "final_model.pt"))
    dt = np.diff(stats["sec"]) if len(stats["sec"]) > 1 else np.array([float("nan")])
    rep = {"arm": a.arm, "seed": a.seed, "iterations_done": int(done), "iterations_requested": a.iterations, "patch": a.patch, "mapping": mapping, "label_set": list(label_set), "trainable_params": int(n_tr), "all_params": int(n_all),
           "final_model_sha256": sha256(os.path.join(out, "final_model.pt")), "weights_sha256": sha256(a.weights) if a.weights else None, "driver_sha256": sha256(os.path.abspath(__file__)), "overlay_sha256": sha256(overlay), "monai": monai.__version__, "torch": torch.__version__,
           "seconds_per_iteration_median": float(np.nanmedian(dt)), "loss_first": stats["loss"][:5], "loss_last": stats["loss"][-5:], "loss_all_finite": bool(np.all(np.isfinite(stats["loss"]))) if stats["loss"] else None,
           "edited_patch_fraction": (float(np.mean(stats["edited"])) if stats["edited"] else None), "n_batches_seen": len(stats["edited"]), "wall_seconds": time.time() - stats["t0"]}
    if a.device.startswith("cuda"):
        rep["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9; rep["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    json.dump(rep, open(os.path.join(out, "run_report.json"), "w"), indent=1); print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
