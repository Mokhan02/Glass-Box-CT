"""Train a BatchTopK SAE on extracted tokens for one hook, then characterise every feature (plan Step 5).
Usage: python scripts/train_sae.py --acts cache/acts --hook mid --expansion 16 --k 32 --epochs 5 --out cache/sae
Scans are split by id (every 10th scan held out for validation), so no scan's tokens are in both sets.
Outputs <out>/<hook>_x<exp>_k<k>/{sae.pt, metrics.json, feature_stats.npz}.
feature_stats: per feature -> firing frequency, mean HU when active, fraction of its activations on each label
(0 = background, 1.. = spleen, liver, aorta, iliopsoas L/R, autochthon L/R, L3), max activation."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from glassbox.sae.batchtopk import BatchTopKSAE, dead_fraction, fit_scale, fvu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--acts", default="cache/acts")
ap.add_argument("--hook", default="mid", choices=["bottleneck", "mid", "decoder_F"])
ap.add_argument("--expansion", type=int, default=16)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--out", default="cache/sae")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-scans", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = torch.device(a.device)
files = sorted(glob.glob(os.path.join(a.acts, "*.npz")))
if a.max_scans:
    files = files[: a.max_scans]
val_files = files[::10]
train_files = [f for f in files if f not in set(val_files)]
print(f"{len(train_files)} train scans, {len(val_files)} val scans", flush=True)


def load(fs):
    xs, hu, lab = [], [], []
    for f in fs:
        z = np.load(f)
        xs.append(z[f"{a.hook}_x"]); hu.append(z[f"{a.hook}_hu"]); lab.append(z[f"{a.hook}_label"])
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(hu).astype(np.float32)),
            torch.from_numpy(np.concatenate(lab).astype(np.int64)))


Xtr, HUtr, Ltr = load(train_files)
Xva, _, _ = load(val_files)
Xtr, Xva = Xtr.to(dev), Xva.to(dev)  # fp16 on the GPU
print(f"tokens: train {len(Xtr):,}  val {len(Xva):,}  d={Xtr.shape[1]}", flush=True)
sae = BatchTopKSAE(Xtr.shape[1], a.expansion, a.k, seed=a.seed).to(dev)
sae.scale.fill_(fit_scale(Xtr[:200_000].float()))
opt = torch.optim.Adam(sae.parameters(), lr=a.lr)
steps_per_epoch = len(Xtr) // a.batch
total = steps_per_epoch * a.epochs
t0, step, hist = time.time(), 0, []
for ep in range(a.epochs):
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(steps_per_epoch):
        x = Xtr[perm[i * a.batch:(i + 1) * a.batch]].float() * sae.scale
        x_hat, _ = sae(x)
        loss = ((x - x_hat) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
        step += 1
        if step % 200 == 0 or step == total:
            hist.append({"step": step, "fvu": fvu(x, x_hat), "dead": dead_fraction(sae, 1000)})
            print(f"ep {ep} step {step}/{total} fvu {hist[-1]['fvu']:.4f} dead {hist[-1]['dead']:.3f} {time.time() - t0:.0f}s", flush=True)

sae.eval()
with torch.no_grad():
    num = den = 0.0
    xv = Xva.float() * sae.scale
    mu = xv.mean(0)
    for i in range(0, len(xv), 16384):
        b = xv[i:i + 16384]
        num += float(((b - sae.decode(sae.encode(b, use_threshold=True))) ** 2).sum())
        den += float(((b - mu) ** 2).sum())
    val_fvu = num / den
    n = sae.n
    count = torch.zeros(n, device=dev); hu_sum = torch.zeros(n, device=dev); mx = torch.zeros(n, device=dev)
    cls = torch.zeros(n, 9, device=dev)
    onehot = torch.nn.functional.one_hot(Ltr.to(dev), 9).float()
    hu = HUtr.to(dev)
    for i in range(0, len(Xtr), 16384):
        z = sae.encode(Xtr[i:i + 16384].float() * sae.scale, use_threshold=True)
        act = (z > 0).float()
        count += act.sum(0); hu_sum += act.T @ hu[i:i + 16384]; cls += act.T @ onehot[i:i + 16384]
        mx = torch.maximum(mx, z.max(0).values)
dead_final = float((count == 0).float().mean())
run = os.path.join(a.out, f"{a.hook}_x{a.expansion}_k{a.k}")
os.makedirs(run, exist_ok=True)
torch.save(sae.state_dict(), os.path.join(run, "sae.pt"))
np.savez(os.path.join(run, "feature_stats.npz"), freq=(count / len(Xtr)).cpu().numpy(),
         mean_hu=(hu_sum / count.clamp_min(1)).cpu().numpy(), class_frac=(cls / count.clamp_min(1)[:, None]).cpu().numpy(),
         max_act=mx.cpu().numpy())
metrics = {"hook": a.hook, "expansion": a.expansion, "k": a.k, "d_in": int(Xtr.shape[1]), "n_features": int(n),
           "train_tokens": int(len(Xtr)), "val_tokens": int(len(Xva)), "train_scans": len(train_files), "val_scans": len(val_files),
           "val_fvu": val_fvu, "val_variance_explained": 1 - val_fvu, "dead_fraction_all_train_tokens": dead_final,
           "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed, "history": hist}
json.dump(metrics, open(os.path.join(run, "metrics.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=1))
