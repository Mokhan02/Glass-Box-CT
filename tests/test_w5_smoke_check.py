import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)


def run(tmp_path, times, losses, edits, arm="BRIGHT"):
    tl = tmp_path / "tl.txt"; tl.write_text("".join(f"x: train_loss {a}\nx: val_loss {b}\nx: Epoch time: {t} s\n" for t, (a, b) in zip(times, losses)))
    el = tmp_path / "el.txt"; el.write_text("".join(f"{arm} {tum} {ap} {d}\n" for tum, ap, d in edits)); out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "w5_smoke_check.py"), arm, str(tl), str(el), str(out)], capture_output=True, text=True)
    return r.returncode, json.load(open(out))


GOOD_E = [(1, 1, "20.5,31.0")] * 50 + [(1, 0, "")] * 50 + [(0, 0, "")] * 30
GOOD_L = [(-0.1, -0.2)] * 5


def test_pass(tmp_path):
    rc, r = run(tmp_path, [130, 61, 60, 62, 61], GOOD_L, GOOD_E); assert rc == 0 and r["pass"]


def test_slow_epochs_fail(tmp_path):
    rc, r = run(tmp_path, [130, 80, 80, 80, 80], GOOD_L, GOOD_E); assert rc == 1 and not r["checks"]["epoch_time_within_10pct"]


def test_nan_loss_fail(tmp_path):
    rc, r = run(tmp_path, [130, 61, 60, 62, 61], GOOD_L[:4] + [(float("nan"), -0.2)], GOOD_E); assert rc == 1 and not r["checks"]["losses_finite"]


def test_inactive_or_out_of_range_fail(tmp_path):
    rc, r = run(tmp_path, [130, 61, 60, 62, 61], GOOD_L, [(1, 0, "")] * 100); assert rc == 1 and not r["checks"]["transform_active"]
    rc, r = run(tmp_path, [130, 61, 60, 62, 61], GOOD_L, [(1, 1, "-20.0")] * 50 + [(1, 0, "")] * 50); assert rc == 1 and not r["checks"]["transform_active"]
