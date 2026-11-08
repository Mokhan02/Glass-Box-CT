"""W4b and W6 pre-registered analysis (results/w4b_waw/PREREG.md, results/w6_vista_finetune/PREREG.md), applied as written. CPU. Frozen and hashed BEFORE any fine-tuned output exists; any later change is a logged deviation.
Input: the paired table (results/w4_waw/paired_instances), and per arm one or more scores CSVs written by scripts/w4_waw_score.py (halves concatenated); the alignment-gate JSON is read from the same folder (alignment_<arm>.json). Arms (names fixed by the PREREGs):
  new: BRIGHT, SHAM, PLAIN, BRIGHT-DEC, BRIGHT-s2 (W6 fine-tuned VISTA-3D), TS (TotalSegmentator liver_vessels, held-out status inferred, not author-confirmed)
  reference, labelled ALREADY SEEN (analysed in W4 before W4b existed): BASELINE (VISTA-3D), A (Universal Model), D0, B50
Output order, as the researcher asked: (1) the W4b PRIMARY (theta(BRIGHT) - theta(SHAM)), (2) its secondaries (BRIGHT vs PLAIN, DEC vs SHAM, NO PENALTY for BRIGHT), (3) per-model W4b labels (reference arms marked ALREADY SEEN), (4) the W6 (c)-type tests on the arterial series (BRIGHT vs SHAM primary label, vs PLAIN, replication with BRIGHT-s2),
(5) the decoder-only ratio R, (6) descriptive items (W4 (a) gaps, v1 contrast, the F2 tumours, penalised conditional logit), (7) the sensitivities with the list of verdicts that change.
Usage: python scripts/w6_analyze.py <paired dir> <out dir> --arm NAME=scores_a.csv[,scores_b.csv] [--arm ...] [--mask-check mask_check_cohort.json] [--b 2000]"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from glassbox.seg import w4_stats as S4  # noqa: E402
from glassbox.seg import w6_stats as S6  # noqa: E402

_spec = importlib.util.spec_from_file_location("w4an", os.path.join(HERE, "w4_waw_analyze.py")); AN = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AN)
SEEN = {"BASELINE", "A", "D0", "B50"}
NEW = ["BRIGHT", "SHAM", "PLAIN", "BRIGHT-DEC", "BRIGHT-s2", "TS"]


def load_arm(paths):
    rows, gate = [], set()
    for p in paths:
        rows += list(csv.DictReader(open(p))); aj = os.path.join(os.path.dirname(p), os.path.basename(p).replace("scores_", "alignment_").replace(".csv", ".json"))
        if os.path.exists(aj):
            gate |= {tuple(c.split("_")) for c in json.load(open(aj))["series_gate_fail"]}
    return rows, gate


def arrays(T, keys):
    ks = sorted(keys); g = lambda f: np.array([f(T[k]) for k in ks])      # noqa: E731
    return {"keys": ks, "pat": g(lambda e: e["pat"]), "det_a": g(lambda e: e["obs"][1]["det"]), "det_p": g(lambda e: e["obs"][2]["det"]), "ca": g(lambda e: e["c"][1]), "cp": g(lambda e: e["c"][2]), "drawn_a": g(lambda e: e["drawn"] == 1),
            "dice_a": g(lambda e: e["obs"][1]["dice"])}


def common(Ts, names):
    return set.intersection(*[set(Ts[n]) for n in names])


def theta_tests(Ts, B):
    """All W4b tests on the tumours common to the arms involved in each test."""
    res = {"models": {}, "primary": None, "secondary": {}}
    for n, T in Ts.items():
        A = arrays(T, T.keys()); f1, nf, f2 = S6.classes(A["ca"], A["cp"]); t = S6.theta_ci(A["pat"], A["det_a"], A["det_p"], f1, nf, A["drawn_a"], f"w4b_theta_{n}", B)
        res["models"][n] = {**t, "label": S6.label_theta(t), "equivalence_label_margin_0.25": S6.label_equivalence(t), "already_seen": n in SEEN, "raw": S6.raw_components(A["det_a"], A["det_p"], f1, nf), "n_F2_excluded": int(f2.sum())}

    def pair(a1, a0, nm):
        if a1 not in Ts or a0 not in Ts:
            return {"label": "NOT RUN (arm missing)"}
        A1 = arrays(Ts[a1], common(Ts, [a1, a0])); A0 = arrays(Ts[a0], common(Ts, [a1, a0])); assert A1["keys"] == A0["keys"]; f1, nf, _ = S6.classes(A1["ca"], A1["cp"])
        r = S6.theta_diff_ci(A1["pat"], (A1["det_a"], A1["det_p"]), (A0["det_a"], A0["det_p"]), f1, nf, A1["drawn_a"], f"w4b_{nm}", B); return {**r, "label": S6.label_reduces(r), "n_tumours": len(A1["keys"])}
    res["primary"] = pair("BRIGHT", "SHAM", "primary_BRIGHT_minus_SHAM"); res["secondary"]["S1_BRIGHT_minus_PLAIN"] = pair("BRIGHT", "PLAIN", "S1_BRIGHT_minus_PLAIN"); res["secondary"]["S2_DEC_minus_SHAM"] = pair("BRIGHT-DEC", "SHAM", "S2_DEC_minus_SHAM")
    res["secondary"]["S3_BRIGHT_NO_PENALTY"] = res["models"].get("BRIGHT", {}).get("equivalence_label_margin_0.25", "NOT RUN (arm missing)"); return res


def c_pair(Ts, a1, a0, B):
    """W6 (c)-type test on the arterial series: gap = dark minus bright by the arterial contrast; Delta = gap(a1) - gap(a0), shared patient resamples, the W5 CLOSES rule per metric."""
    if a1 not in Ts or a0 not in Ts:
        return {"label": "NOT RUN (arm missing)"}
    ks = sorted(common(Ts, [a1, a0])); pat = [Ts[a1][k]["pat"] for k in ks]; br = [Ts[a1][k]["c"][1] > 0 for k in ks]; out = {"n_tumours": len(ks)}; v = {}
    vals = {m: {a: [Ts[a][k]["obs"][1][m] for k in ks] for a in (a1, a0)} for m in ("det", "dice")}
    for m in ("det", "dice"):
        r = S4.paired_delta(pat, br, vals[m], a1, a0, f"w6_c_{a1}_{a0}_{m}", B); r["verdict"] = S4.closes_verdict(r["delta"], r["lo"], r["hi"]); out[m] = r; v[(a0, m)] = r["verdict"]
    out["label"] = S4.overall_closes(v); out["bright_arterial_detection"] = {a: float(np.mean([vals["det"][a][i] for i, b in enumerate(br) if b])) for a in (a1, a0)}; return out


def w6_tests(Ts, B):
    res = {"BRIGHT_vs_SHAM (PRIMARY, W6)": c_pair(Ts, "BRIGHT", "SHAM", B), "BRIGHT_vs_PLAIN": c_pair(Ts, "BRIGHT", "PLAIN", B), "BRIGHT_vs_BASELINE (descriptive, confounded with fine-tuning)": c_pair(Ts, "BRIGHT", "BASELINE", B),
           "BRIGHT-s2_vs_SHAM (replication)": c_pair(Ts, "BRIGHT-s2", "SHAM", B), "DEC_vs_SHAM": c_pair(Ts, "BRIGHT-DEC", "SHAM", B)}
    if all(a in Ts for a in ("PLAIN", "BRIGHT-DEC", "BRIGHT")):
        ks = sorted(common(Ts, ["PLAIN", "BRIGHT-DEC", "BRIGHT"])); pat = [Ts["BRIGHT"][k]["pat"] for k in ks]; br = [Ts["BRIGHT"][k]["c"][1] > 0 for k in ks]
        res["Q2_decoder_only_ratio"] = {m: S6.gap_ratio(pat, br, [Ts["PLAIN"][k]["obs"][1][m] for k in ks], [Ts["BRIGHT-DEC"][k]["obs"][1][m] for k in ks], [Ts["BRIGHT"][k]["obs"][1][m] for k in ks], f"w6_q2_{m}", B) for m in ("det", "dice")}
    else:
        res["Q2_decoder_only_ratio"] = "NOT RUN (arm missing)"
    if all(a in Ts for a in ("BRIGHT", "BRIGHT-s2")):
        ks = sorted(common(Ts, ["BRIGHT", "BRIGHT-s2"])); pat = [Ts["BRIGHT"][k]["pat"] for k in ks]; br = [Ts["BRIGHT"][k]["c"][1] > 0 for k in ks]
        res["seed_difference_of_gaps (BRIGHT vs BRIGHT-s2, arterial)"] = {m: S4.paired_delta(pat, br, {a: [Ts[a][k]["obs"][1][m] for k in ks] for a in ("BRIGHT", "BRIGHT-s2")}, "BRIGHT", "BRIGHT-s2", f"w6_seed_{m}", B) for m in ("det", "dice")}
    return res


def descriptive(Ts, o, b_logit):
    out = {}
    for n, T in Ts.items():
        A = arrays(T, T.keys()); ba, bp = (A["ca"] > 0).astype(float), (A["cp"] > 0).astype(float); D = np.column_stack([ba - bp, np.abs(A["ca"]) / 10 - np.abs(A["cp"]) / 10, np.array([T[k]["obs"][1]["prop"] - T[k]["obs"][2]["prop"] for k in A["keys"]], float)])
        out[n] = {"a_gap": AN.family_a(T, n, o), "v1_contrast": AN.effective_n(T, n), "F2_tumours": AN.reverse_flippers(T), "conditional_logit_penalised": S6.cond_logit_penalised(A["pat"], A["det_a"], A["det_p"], D, ["sign", "abs_contrast", "is_propagated"], f"w4b_cl_{n}", b_logit), "already_seen": n in SEEN}
    return out


def run(tum, arms, o, B, full=True):
    Ts = {n: AN.build(tum, r, g, o) for n, (r, g) in arms.items()}; res = {"n_tumours": {n: len(T) for n, T in Ts.items()}, "w4b": theta_tests(Ts, B), "w6": w6_tests(Ts, B)}
    verd = {"w4b/PRIMARY": res["w4b"]["primary"].get("label")}
    for k, v in res["w4b"]["secondary"].items():
        verd[f"w4b/{k}"] = v["label"] if isinstance(v, dict) else v
    for n, m in res["w4b"]["models"].items():
        verd[f"w4b/model/{n}"] = m["label"]
    for k, v in res["w6"].items():
        if isinstance(v, dict) and "label" in v:
            verd[f"w6/{k}"] = v["label"]
    if isinstance(res["w6"].get("Q2_decoder_only_ratio"), dict):
        for m, v in res["w6"]["Q2_decoder_only_ratio"].items():
            verd[f"w6/Q2/{m}"] = v["label"]
    if full:
        res["descriptive"] = descriptive(Ts, o, max(200, B // 2))
    return res, verd


def main(argv):
    opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--arm", "--mask-check", "--b"))]; paired, out = pos[0], pos[1]; os.makedirs(out, exist_ok=True); B = int(opt("--b", S6.B_TEST))
    arms = {}
    for i, a in enumerate(argv):
        if a == "--arm":
            name, paths = argv[i + 1].split("=", 1); arms[name] = load_arm(paths.split(","))
    tum = {(r["patient"], r["mask_phase"], r["rank"]): r for r in csv.DictReader(open(os.path.join(paired, "tumours_paired.csv")))}
    base = {"cats": {"K"}, "det": "detected", "excl_pat": set(), "top2": False, "min_diam": 0.0, "gate": False, "contrast": ""}; mc = opt("--mask-check"); v2 = set()
    if mc and os.path.exists(mc):
        v2 = {t["file"].split("_")[0] for e in json.load(open(mc))["patients"].values() for t in e["tumour"] if not t["values_exactly_0_1"]}
    primary, pv = run(tum, arms, base, B); json.dump(primary, open(os.path.join(out, "primary.json"), "w"), indent=1, default=float)
    sens = {"dilation_2mm_propagated": {"det": "detected_dil2"}, "top2_per_series": {"top2": True}, "excl_value2_patients": {"excl_pat": v2}, "incl_Q": {"cats": {"K", "Q"}}, "excl_alignment_gate_failures": {"gate": True}, "min_diam_15mm": {"min_diam": 15.0},
            "liver_eroded_3mm": {"contrast": "_eroded"}, "shell_median": {"contrast": "_median"}}
    table = {}
    for name, ch in sens.items():
        r, vv = run(tum, arms, {**base, **ch}, B, full=False); table[name] = {"verdicts": vv, "changed": {k: {"primary": pv[k], "sensitivity": vv.get(k)} for k in pv if vv.get(k) != pv[k]}}
    json.dump({"primary_verdicts": pv, "sensitivities": table}, open(os.path.join(out, "sensitivity_table.json"), "w"), indent=1, default=float)
    P = primary["w4b"]; L = ["# W4b / W6 analysis (PREREGs applied as written)", "", "## 1. W4b PRIMARY: theta(BRIGHT) - theta(SHAM) (one-sided alpha 0.025)", "", f"{json.dumps(P['primary'], default=float)}", "", "## 2. W4b secondaries", ""]
    L += [f"- {k}: {json.dumps(v, default=float)}" for k, v in P["secondary"].items()] + ["", "## 3. Per-model W4b labels (ALREADY SEEN = analysed in W4 before W4b)", ""]
    L += [f"- {n}{' (ALREADY SEEN)' if m['already_seen'] else ''}: theta {m['theta']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}] -> {m['label']}; NO PENALTY test (margin 0.25): {m['equivalence_label_margin_0.25']}; F1 {m['n_F1']}, NF {m['n_NF']}" for n, m in P["models"].items()]
    L += ["", "## 4. W6 (c)-type tests on the arterial series", ""] + [f"- {k}: {v.get('label') if isinstance(v, dict) else v}" for k, v in primary["w6"].items()] + ["", "## 7. Sensitivities: verdicts that change against the primary", ""]
    L += [f"- {n}: " + ("no verdict changes" if not t["changed"] else "; ".join(f"{k}: {c['primary']} -> {c['sensitivity']}" for k, c in t["changed"].items())) for n, t in table.items()]
    open(os.path.join(out, "ANALYSIS.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L[:30]))


if __name__ == "__main__":
    main(sys.argv[1:])
