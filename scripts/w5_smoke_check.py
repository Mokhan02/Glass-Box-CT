"""Check one W5 smoke run. Usage: python scripts/w5_smoke_check.py <arm BRIGHT|SHAM> <training_log> <edit_log> <out json> [reference s/epoch, default 61]
PASS needs: 5 epochs logged; every train/val loss finite; the mean time of epochs 2 to 5 (epoch 1 includes torch.compile warm-up) within 10% of the reference; the transform active (edit log non-empty, share of
tumour-containing samples edited within 0.35 to 0.65 for p = 0.5, every logged delta inside the arm's range)."""
import json
import math
import re
import sys

arm, tl, el, out = sys.argv[1:5]; ref = float(sys.argv[5]) if len(sys.argv) > 5 else 61.0
txt = open(tl).read()
times = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", txt)]
tr = [float(x) for x in re.findall(r"train_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]; va = [float(x) for x in re.findall(r"val_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]
lo, hi = (10.0, 60.0) if arm == "BRIGHT" else (-60.0, -10.0)
n_tum = n_ed = n_all = 0; bad = 0; nd = 0
for line in open(el):
    f = line.split(" ")
    if len(f) < 3:
        continue
    n_all += 1; n_tum += int(f[1]); n_ed += int(f[1]) * int(f[2])
    for d in (f[3].strip().split(",") if len(f) > 3 and f[3].strip() else []):
        nd += 1; bad += not (lo - 1e-6 <= float(d) <= hi + 1e-6)
mean25 = sum(times[1:5]) / 4 if len(times) >= 5 else float("nan")
res = {"arm": arm, "epochs_logged": len(times), "epoch_times_s": times, "mean_epoch_2_to_5_s": mean25, "reference_s": ref, "train_loss": tr, "val_loss": va, "samples_logged": n_all, "samples_with_tumour": n_tum, "edit_share_among_tumour_samples": (n_ed / n_tum if n_tum else None), "deltas_logged": nd, "deltas_out_of_range": bad}
res["checks"] = {"five_epochs": len(times) == 5 and len(tr) == 5 and len(va) == 5, "losses_finite": bool(tr and va and all(math.isfinite(x) for x in tr + va)), "epoch_time_within_10pct": abs(mean25 - ref) <= 0.10 * ref if not math.isnan(mean25) else False,
                 "transform_active": n_all > 0 and n_tum > 0 and n_ed > 0 and (0.35 <= n_ed / n_tum <= 0.65) and nd > 0 and bad == 0}
res["pass"] = all(res["checks"].values()); json.dump(res, open(out, "w"), indent=1); print(("PASS" if res["pass"] else "FAIL"), json.dumps(res["checks"]), f"mean epoch 2-5 {mean25:.1f}s", f"edit share {res['edit_share_among_tumour_samples']}")
sys.exit(0 if res["pass"] else 1)
