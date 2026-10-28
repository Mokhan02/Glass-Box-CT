"""W5 Amendment 2 early check (2-epoch smoke of BRIGHT-50-s2). Usage: python scripts/w5_early_check.py <expected seed> <training_log> <edit_log> <out json> [reference s/epoch, default 61]
PASS needs: the seed in the log equals the expected seed; the log states the BRIGHT augmentation with p = 0.5; 2 epochs logged with finite train and validation losses; epoch 2 time (epoch 1 includes torch.compile warm-up) within 10% of the reference;
the transform active: the edit log is non-empty and the share of tumour-containing samples edited is in 0.35 to 0.65, every logged delta inside +10 to +60 HU."""
import json
import math
import re
import sys

seed, tl, el, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; ref = float(sys.argv[5]) if len(sys.argv) > 5 else 61.0
txt = open(tl).read(); m = re.search(r"W5: seed (\d+); augmentation (\w+) p=([0-9.]+)", txt)
times = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", txt)]
tr = [float(x) for x in re.findall(r"train_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]; va = [float(x) for x in re.findall(r"val_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]
n_tum = n_ed = n_all = nd = bad = 0
for line in open(el):
    f = line.split(" ")
    if len(f) < 3:
        continue
    n_all += 1; n_tum += int(f[1]); n_ed += int(f[1]) * int(f[2])
    for d in (f[3].strip().split(",") if len(f) > 3 and f[3].strip() else []):
        nd += 1; bad += not (10 - 1e-6 <= float(d) <= 60 + 1e-6)
share = n_ed / n_tum if n_tum else None
res = {"expected_seed": seed, "log_seed": int(m.group(1)) if m else None, "log_arm": m.group(2) if m else None, "log_p": float(m.group(3)) if m else None, "epochs_logged": len(times), "epoch_times_s": times, "train_loss": tr, "val_loss": va,
       "samples_logged": n_all, "samples_with_tumour": n_tum, "edit_share_among_tumour_samples": share, "deltas_logged": nd, "deltas_out_of_range": bad, "reference_s": ref}
res["checks"] = {"seed_in_log": bool(m) and int(m.group(1)) == seed, "augmentation_stated": bool(m) and m.group(2) == "BRIGHT" and float(m.group(3)) == 0.5, "two_epochs": len(times) == 2 and len(tr) == 2 and len(va) == 2,
                 "losses_finite": bool(tr and va and all(math.isfinite(x) for x in tr + va)), "epoch2_time_within_10pct": len(times) >= 2 and abs(times[1] - ref) <= 0.10 * ref,
                 "transform_active": n_all > 0 and share is not None and 0.35 <= share <= 0.65 and nd > 0 and bad == 0}
res["pass"] = all(res["checks"].values()); json.dump(res, open(out, "w"), indent=1)
print("PASS" if res["pass"] else "FAIL", json.dumps(res["checks"]), "seed", res["log_seed"], "epoch2", times[1] if len(times) > 1 else None, "edit share", share)
sys.exit(0 if res["pass"] else 1)
