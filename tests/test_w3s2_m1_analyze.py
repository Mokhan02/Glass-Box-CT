import json
import os
import subprocess
import sys

import numpy as np


def _tokens(path, n_cases=15, vista=False, signal=2.5, seed=1):
    """Synthetic token files: tumour tokens separable from liver at `mid`; at decoder_F the missed tumour tokens are NOT separable (lost late); raw HU carries no signal."""
    rng = np.random.default_rng(seed); os.makedirs(path, exist_ok=True); cases = [f"c{i:02d}" for i in range(n_cases)]
    for c in cases:
        z = {"gt_tumour_mm3": float(rng.uniform(1000, 9000)), "logit_fn_n": 50, "logit_tp_n": 50, "logit_fn": rng.normal(-3.0, 1.0, 50).astype(np.float16), "logit_tp": rng.normal(2.0, 1.0, 50).astype(np.float16)}
        for h, d in (("mid", 16), ("bottleneck", 12), ("decoder_F", 8)):
            n_t, n_l = 60, 140; is_t = np.r_[np.ones(n_t, bool), np.zeros(n_l, bool)]; pc = np.r_[rng.choice([0.0, 0.9], n_t, p=[0.5, 0.5]), np.zeros(n_l)]; edt = np.r_[rng.uniform(0, 8, n_t), np.full(n_l, 20.0)]
            fn = is_t & (pc < 0.5); x = rng.normal(0, 1, (n_t + n_l, d)).astype(np.float32)
            if h == "mid":
                x[is_t, 0] += signal
            if h == "decoder_F":
                x[is_t & (pc >= 0.5), 0] += signal; x[fn, 0] += 0.0       # only detected tumour tokens are separable late
            z.update({f"{h}_x": x.astype(np.float16), f"{h}_xr": rng.normal(0, 1, x.shape).astype(np.float16), f"{h}_xyz": rng.integers(0, 100, (n_t + n_l, 3)).astype(np.int16), f"{h}_is_tum": is_t, f"{h}_pred_cov": pc.astype(np.float16),
                      f"{h}_edt": edt.astype(np.float16), f"{h}_hu14": rng.normal(0, 1, (n_t + n_l, 14)).astype(np.float16)})
        np.savez_compressed(os.path.join(path, c + ".npz"), **z)
    return cases


def _run(args):
    return subprocess.run([sys.executable, "scripts/w3s2_m1_analyze.py", *args], capture_output=True, text=True, check=True)


def test_synthetic_lost_late_and_arm_seed_independence(tmp_path):
    cases = _tokens(str(tmp_path / "tok" / "tokens")); open(tmp_path / "cases.txt", "w").write("\n".join(cases))
    _run(["--tokens", str(tmp_path / "tok"), "--cases", str(tmp_path / "cases.txt"), "--out", str(tmp_path / "o1"), "--arm", "synthetic_A", "--n-boot", "200"])
    d = json.load(open(tmp_path / "o1" / "decision.json"))["primary"]
    assert d["verdict"] == "LOST LATE" and d["represents"] and d["R_enc"] >= 0.5 and d["R_hu"] < 0.3 and d["late_criterion_b"]
    _run(["--tokens", str(tmp_path / "tok"), "--cases", str(tmp_path / "cases.txt"), "--out", str(tmp_path / "o2"), "--arm", "synthetic_A", "--n-boot", "200"])
    assert json.load(open(tmp_path / "o2" / "decision.json"))["primary"] == d                      # same arm name -> identical numbers
    _run(["--tokens", str(tmp_path / "tok"), "--cases", str(tmp_path / "cases.txt"), "--out", str(tmp_path / "o3"), "--arm", "no_signal_placeholder", "--n-boot", "200"])
    assert json.load(open(tmp_path / "o3" / "decision.json"))["seed"] != json.load(open(tmp_path / "o1" / "decision.json"))["seed"]


def test_never_represented_without_signal(tmp_path):
    cases = _tokens(str(tmp_path / "tok" / "tokens"), signal=0.0, seed=2); open(tmp_path / "cases.txt", "w").write("\n".join(cases))
    _run(["--tokens", str(tmp_path / "tok"), "--cases", str(tmp_path / "cases.txt"), "--out", str(tmp_path / "o"), "--arm", "null", "--n-boot", "200"])
    assert json.load(open(tmp_path / "o" / "decision.json"))["primary"]["verdict"] in ("NEVER REPRESENTED", "INCONCLUSIVE")


def test_vista_compat_layout_runs_and_checks_against_itself(tmp_path):
    for m in ("trained", "random"):
        _tokens(str(tmp_path / "vp" / m), seed=3 if m == "trained" else 4)
    _run(["--vista-compat", "--tokens", str(tmp_path / "vp"), "--out", str(tmp_path / "v1"), "--arm", "vista", "--seed", "0", "--n-boot", "200"])
    _run(["--vista-compat", "--tokens", str(tmp_path / "vp"), "--out", str(tmp_path / "v2"), "--arm", "vista", "--seed", "0", "--n-boot", "200", "--check", str(tmp_path / "v1" / "probe_table.csv")])
    r = json.load(open(tmp_path / "v2" / "reproduction_check.json")); assert r["reproduced"] and r["compared_values"] > 50
