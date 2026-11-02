"""W5 Amendment 4 early check (2-epoch smoke of BRIGHT-25 or D0-seed2). Usage: python scripts/w5a4_early_check.py <B25|D0S2> <training_log> <edit_log> <out json> [reference s/epoch, default 61]
Common PASS criteria: the expected seed (B25 20261020, D0S2 20261121) in the log; the arm line `W5: seed S; augmentation ARM p=P` as expected (B25: BRIGHT, 0.25; D0S2: None, 0.0); 2 epochs logged with finite train and validation losses;
epoch 2 time (epoch 1 includes torch.compile warm-up) within 10% of the reference. B25 additionally: the transform active, i.e. the edit log is non-empty, the share of tumour-containing samples edited is in 0.15 to 0.35 (p = 0.25 times the measured application fraction of about 0.9),
and every logged delta is inside +10 to +60 HU. D0S2 additionally: no augmentation, i.e. the edit log is missing or has no edited sample."""
import json
import math
import os
import re
import sys

EXPECT = {"B25": (20261020, "BRIGHT", 0.25), "D0S2": (20261121, "None", 0.0)}
tag, tl, el, out = sys.argv[1:5]; ref = float(sys.argv[5]) if len(sys.argv) > 5 else 61.0; seed, arm, p = EXPECT[tag]
txt = open(tl).read(); m = re.search(r"W5: seed (\d+); augmentation (\w+) p=([0-9.]+)", txt)
times = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", txt)]
tr = [float(x) for x in re.findall(r"train_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]; va = [float(x) for x in re.findall(r"val_loss (-?[0-9.]+(?:e-?[0-9]+)?|nan|-?inf)", txt)]
n_tum = n_ed = n_all = nd = bad = 0
if os.path.exists(el):
    for line in open(el):
        f = line.split(" ")
        if len(f) < 3:
            continue
        n_all += 1; n_tum += int(f[1]); n_ed += int(f[1]) * int(f[2])
        for d in (f[3].strip().split(",") if len(f) > 3 and f[3].strip() else []):
            nd += 1; bad += not (10 - 1e-6 <= float(d) <= 60 + 1e-6)
share = n_ed / n_tum if n_tum else None
res = {"arm_tag": tag, "expected_seed": seed, "log_seed": int(m.group(1)) if m else None, "log_arm": m.group(2) if m else None, "log_p": float(m.group(3)) if m else None, "epochs_logged": len(times), "epoch_times_s": times, "train_loss": tr, "val_loss": va,
       "samples_logged": n_all, "samples_with_tumour": n_tum, "samples_edited": n_ed, "edit_share_among_tumour_samples": share, "deltas_logged": nd, "deltas_out_of_range": bad, "reference_s": ref}
checks = {"seed_in_log": bool(m) and int(m.group(1)) == seed, "arm_line_as_expected": bool(m) and m.group(2) == arm and float(m.group(3)) == p, "two_epochs": len(times) == 2 and len(tr) == 2 and len(va) == 2,
          "losses_finite": bool(tr and va and all(math.isfinite(x) for x in tr + va)), "epoch2_time_within_10pct": len(times) >= 2 and abs(times[1] - ref) <= 0.10 * ref}
if tag == "B25":
    checks["transform_active"] = n_all > 0 and share is not None and 0.15 <= share <= 0.35 and nd > 0 and bad == 0
else:
    checks["no_augmentation"] = n_ed == 0 and nd == 0
res["checks"] = checks; res["pass"] = all(checks.values()); json.dump(res, open(out, "w"), indent=1)
print("PASS" if res["pass"] else "FAIL", json.dumps(checks), "seed", res["log_seed"], "epoch2", times[1] if len(times) > 1 else None, "edit share", share)
sys.exit(0 if res["pass"] else 1)
