"""W4 (WAW-TACE) pre-registered analysis (results/w4_waw/PREREG.md), applied as written. CPU.
Inputs: paired dir (tumours_paired.csv, patient_qa.csv from scripts/w4_waw_paired_contrast.py), score dirs holding scores_<arm>.csv / alignment_<arm>.json (scripts/w4_waw_score.py; several dirs = the two instances are concatenated),
arms: vista (VISTA-3D class 26), D0, A (Universal Model), B50, B25. Optional mask-check json (patients with value-2 tumour files).
Families, never pooled: (a) replication per model on the drawn phase (+ strata by drawn phase); (c) D0 vs BRIGHT-50 (verdict) and BRIGHT-25 (descriptive) on the arterial series; (b) conditional logistic regression, sign OR per model, plus the
5 A-dark/P-bright flippers per model; then the sensitivities (each recomputes the verdicts; a table lists any verdict that changes against the primary).
Usage: python scripts/w4_waw_analyze.py <paired dir> <out dir> <scores dir> [<scores dir> ...] [--mask-check mask_check_cohort.json] [--host1-paired <host1 paired dir>] [--b-scale 1.0]"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.seg import w4_stats as S  # noqa: E402

ARMS = ["vista", "D0", "A", "B50", "B25"]
B_MODELS = ["vista", "D0", "A", "B50"]      # (b) verdict models; B25 descriptive
CONTRAST_COLS = ["c_drawn", "c_other", "c_arterial", "c_portal", "c_drawn_eroded", "c_other_eroded", "c_arterial_eroded", "c_portal_eroded", "c_drawn_median", "c_other_median", "c_arterial_median", "c_portal_median"]
FLT = lambda x: float(x) if x not in ("", None) else None      # noqa: E731


def load(paired, score_dirs):
    tum = {}
    for r in csv.DictReader(open(os.path.join(paired, "tumours_paired.csv"))):
        tum[(r["patient"], r["mask_phase"], r["rank"])] = r
    sc = {}
    for a in ARMS:
        rows = []; gate = set()
        for d in score_dirs:
            p = os.path.join(d, f"scores_{a}.csv")
            if os.path.exists(p):
                rows += list(csv.DictReader(open(p)))
                g = json.load(open(os.path.join(d, f"alignment_{a}.json"))); gate |= {tuple(c.split("_")) for c in g["series_gate_fail"]}
        if rows:
            sc[a] = (rows, gate)
    return tum, sc


def build(tum, rows, gate, o):
    """Tumour-level table for one arm and options: {tkey: {pat, drawn, prop_a, ca, cp, diam, obs{1: dict, 2: dict}}} keeping tumours with both series scored."""
    cs = o.get("contrast", "")      # '' | '_eroded' | '_median'
    col = {1: "c_arterial" + cs, 2: "c_portal" + cs}; out = {}
    for r in rows:
        k = (r["patient"], r["drawn_phase"], r["rank_drawn_phase"]); t = tum.get(k)
        if t is None or t["category"] not in o["cats"] or r[o["det"]] == "" or r["patient"] in o["excl_pat"]:
            continue
        if o["top2"] and int(r["rank_drawn_phase"]) > 2 or float(t["diam_mm"]) < o["min_diam"]:
            continue
        sp = int(r["series_phase"])
        if o["gate"] and (r["patient"], str(sp)) in gate:
            continue
        e = out.setdefault(k, {"pat": r["patient"], "drawn": int(r["drawn_phase"]), "diam": float(t["diam_mm"]), "obs": {}, "c": {1: FLT(t[col[1]]), 2: FLT(t[col[2]])}})
        e["obs"][sp] = {"det": int(r[o["det"]]), "dice": float(r["dice_full"]), "dice10": float(r["dice_within_10mm"]), "cov": float(r["coverage"]), "prop": int(r["is_propagated"]), "diam": float(r["diam_mm_series"])}
    return {k: v for k, v in out.items() if v["c"][1] is not None and v["c"][2] is not None and set(v["obs"]) == {1, 2}}


def family_a(T, arm, o):
    """Replication on the drawn phase: gap = dark minus bright by the drawn-phase contrast; strata by drawn phase."""
    res = {}
    for name, sel in (("all", lambda e: True), ("arterial_drawn", lambda e: e["drawn"] == 1), ("portal_drawn", lambda e: e["drawn"] == 2)):
        es = [e for e in T.values() if sel(e)]
        if not es:
            res[name] = {"label": "INCONCLUSIVE (no tumours)"}; continue
        pat = [e["pat"] for e in es]; br = [e["c"][e["drawn"]] > 0 for e in es]; ob = [e["obs"][e["drawn"]] for e in es]
        d = S.gap_ci(pat, br, [x["det"] for x in ob], f"w4_a_{arm}_{name}_det"); q = S.gap_ci(pat, br, [x["dice"] for x in ob], f"w4_a_{arm}_{name}_dice")
        res[name] = {"detection_gap": d, "dice_gap": q, "detection_label": S.gap_label(d["gap"], d["lo"], d["hi"]), "dice_label": S.gap_label(q["gap"], q["lo"], q["hi"]), "bright_share": float(np.mean(br))}
        res[name]["label"] = S.model_gap_label(res[name]["detection_label"], res[name]["dice_label"])
    return res


def family_c(Ts, o):
    """Arterial series, bright by arterial contrast, same tumours for all arms (intersection). B50 vs D0: verdict; B25 vs D0 and vs B50: descriptive."""
    if "D0" not in Ts or "B50" not in Ts:
        return {"label": "NOT RUN (arm missing)"}
    keys = sorted(set.intersection(*[set(Ts[a]) for a in ("D0", "B50", "B25") if a in Ts])); pat = [Ts["D0"][k]["pat"] for k in keys]; br = [Ts["D0"][k]["c"][1] > 0 for k in keys]; res = {"n_tumours": len(keys)}
    vals = {m: {a: [Ts[a][k]["obs"][1][m] for k in keys] for a in Ts if a in ("D0", "B50", "B25")} for m in ("det", "dice")}
    for a in vals["det"]:
        res[f"gap_{a}"] = {m: S.gap_ci(pat, br, vals[m][a], f"w4_c_{a}_{m}") for m in ("det", "dice")}
    v = {}
    for m in ("det", "dice"):
        r = S.paired_delta(pat, br, vals[m], "B50", "D0", f"w4_c_B50_D0_{m}"); r["verdict"] = S.closes_verdict(r["delta"], r["lo"], r["hi"]); res[f"B50_minus_D0_{m}"] = r; v[("D0", m)] = r["verdict"]
        if "B25" in vals[m]:
            res[f"B25_minus_D0_{m}"] = S.paired_delta(pat, br, vals[m], "B25", "D0", f"w4_c_B25_D0_{m}"); res[f"B25_minus_B50_{m}"] = S.paired_delta(pat, br, vals[m], "B25", "B50", f"w4_c_B25_B50_{m}")
    res["label"] = S.overall_closes(v)
    bk = [i for i, b in enumerate(br) if b]
    if bk:
        res["bright_arterial_detection"] = {a: float(np.mean([vals["det"][a][i] for i in bk])) for a in vals["det"]}
        pb = [pat[i] for i in bk]; res["bright_arterial_detection_B50_minus_D0"] = S.paired_mean_diff(pb, [vals["det"]["B50"][i] for i in bk], [vals["det"]["D0"][i] for i in bk], "w4_c_bright_det")
    return res


def design(T, log_d=True, mag=True, signed=False, dead=0.0):
    """Within-tumour differences (arterial minus portal-venous) for the conditional logit."""
    keys = sorted(T); pat = [T[k]["pat"] for k in keys]; keep = [k for k in keys if min(abs(T[k]["c"][1]), abs(T[k]["c"][2])) >= dead]; cols = []; names = []
    sg = lambda k, ph: float(T[k]["c"][ph] > 0)      # noqa: E731
    f = lambda fn: [fn(k, 1) - fn(k, 2) for k in keep]      # noqa: E731
    if signed:
        cols.append(f(lambda k, ph: T[k]["c"][ph] / 10)); names.append("signed_contrast")
    else:
        cols.append(f(sg)); names.append("sign")
        if mag:
            cols.append(f(lambda k, ph: abs(T[k]["c"][ph]) / 10)); names.append("abs_contrast")
    cols.append(f(lambda k, ph: float(T[k]["obs"][ph]["prop"]))); names.append("is_propagated")
    if log_d:
        cols.append(f(lambda k, ph: math.log(max(T[k]["obs"][ph]["diam"], 1.0)))); names.append("log_diameter")
    live = [j for j, c in enumerate(cols) if names[j] in ("sign", "signed_contrast") or any(abs(v) > 1e-12 for v in c)]      # a covariate identical on both series of every tumour (e.g. log diameter of a drawn mask copied exactly) carries no information
    cols = [cols[j] for j in live]; names = [names[j] for j in live]
    return keep, np.array([T[k]["pat"] for k in keep]), np.array([T[k]["obs"][1]["det"] for k in keep]), np.array([T[k]["obs"][2]["det"] for k in keep]), np.column_stack(cols), names


def family_b(T, arm, o, variant="primary", b=S.B_B):
    kw = {"primary": {}, "no_logd": {"log_d": False}, "no_mag": {"mag": False}, "signed": {"signed": True, "mag": False}, "deadzone5": {"dead": 5.0}}[variant]
    keep, pat, da, dp, D, names = design(T, **kw)
    if len(keep) < 20:
        return {"status": "NOT ESTIMABLE", "label": "INCONCLUSIVE (too few tumours)"}
    r = S.cond_logit(pat, da, dp, D, names, f"w4_b_{arm}" + ("" if variant == "primary" else "_" + variant), b=b); col = "signed_contrast" if variant == "signed" else "sign"
    r["label"] = S.sign_verdict(r, col)
    if r.get("status", "").startswith("OK"):
        ca = np.array([T[k]["c"][1] > 0 for k in keep]); cp = np.array([T[k]["c"][2] > 0 for k in keep]); d = da != dp
        r["discordant_non_flip"] = int((d & (ca == cp)).sum()); r["discordant_flip_A_bright_P_dark"] = int((d & ca & ~cp).sum()); r["discordant_flip_A_dark_P_bright"] = int((d & ~ca & cp).sum())
        r["tumours_flip_A_bright_P_dark"] = int((ca & ~cp).sum()); r["tumours_flip_A_dark_P_bright"] = int((~ca & cp).sum())
    return r


def reverse_flippers(T):
    """The A-dark/P-bright tumours, descriptively (the only ones that break the collinearity of phase and sign)."""
    return [{"patient": e["pat"], "drawn_phase": e["drawn"], "diam_mm": e["diam"], "c_arterial": e["c"][1], "c_portal": e["c"][2], "detected_arterial": e["obs"][1]["det"], "detected_portal": e["obs"][2]["det"], "coverage_arterial": e["obs"][1]["cov"], "coverage_portal": e["obs"][2]["cov"]}
            for e in T.values() if e["c"][1] <= 0 < e["c"][2]]


def effective_n(T, arm, delta=5.0, rho=0.5):
    d = []; pats = []
    for e in T.values():
        a, p = e["c"][1], e["c"][2]
        if a * p < 0 and abs(a) >= delta and abs(p) >= delta and min(abs(a), abs(p)) / max(abs(a), abs(p)) >= rho:
            br, dk = (1, 2) if a > 0 else (2, 1); d.append(e["obs"][br]["det"] - e["obs"][dk]["det"]); pats.append(e["pat"])
    return S.effective_n_design(pats, d, f"w4_eff_{arm}")


def run(tum, sc, o, b_scale=1.0, full=True):
    """All families under options o. Returns the result tree and the verdict summary {name: label}."""
    Ts = {a: build(tum, sc[a][0], sc[a][1], o) for a in sc}; res = {"n_tumours": {a: len(Ts[a]) for a in Ts}, "a": {}, "b": {}}; verd = {}
    for a in Ts:
        res["a"][a] = family_a(Ts[a], a, o); verd[f"a/{a}"] = res["a"][a]["all"]["label"]
        for st in ("arterial_drawn", "portal_drawn"):
            verd[f"a/{a}/{st}"] = res["a"][a][st]["label"]
    res["c"] = family_c(Ts, o); verd["c/B50_vs_D0"] = res["c"]["label"]
    for a in Ts:
        res["b"][a] = family_b(Ts[a], a, o, "primary", int(S.B_B * b_scale)); verd[f"b/{a}"] = res["b"][a]["label"]
    if full:
        res["b_variants"] = {a: {v: family_b(Ts[a], a, o, v, int(S.B_B * b_scale)) for v in ("no_logd", "no_mag", "signed", "deadzone5")} for a in Ts}
        res["effective_n_v1"] = {a: effective_n(Ts[a], a) for a in Ts}; res["reverse_flippers"] = {a: reverse_flippers(Ts[a]) for a in Ts}
    return res, verd


def main(argv):
    opt = lambda n: argv[argv.index(n) + 1] if n in argv else None      # noqa: E731
    mc, h1 = opt("--mask-check"), opt("--host1-paired"); bs = float(opt("--b-scale") or 1.0)
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--mask-check", "--b-scale", "--host1-paired"))]; paired, out, dirs = pos[0], pos[1], pos[2:]; os.makedirs(out, exist_ok=True)
    tum, sc = load(paired, dirs); base = {"cats": {"K"}, "det": "detected", "excl_pat": set(), "top2": False, "min_diam": 0.0, "gate": False, "contrast": ""}
    v2 = set()
    if mc and os.path.exists(mc):
        v2 = {t["file"].split("_")[0] for e in json.load(open(mc))["patients"].values() for t in e["tumour"] if not t["values_exactly_0_1"]}
    primary, pv = run(tum, sc, base, bs); json.dump(primary, open(os.path.join(out, "primary.json"), "w"), indent=1, default=float)
    sens = {"dilation_2mm_propagated": {"det": "detected_dil2"}, "top2_per_series": {"top2": True}, "excl_value2_patients": {"excl_pat": v2}, "incl_Q": {"cats": {"K", "Q"}}, "excl_alignment_gate_failures": {"gate": True},
            "min_diam_15mm": {"min_diam": 15.0}, "liver_eroded_3mm": {"contrast": "_eroded"}, "shell_median": {"contrast": "_median"}}
    table = {}; sens_res = {}
    if h1:      # pre-specified (DEVIATIONS 18): the scouting-run (host1) contrasts for the tumours present in both tables; masks and detections unchanged
        ref = {(r["patient"], r["mask_phase"], r["rank"]): r for r in csv.DictReader(open(os.path.join(h1, "tumours_paired.csv")))}
        swap = {k: ({**v, **{c: ref[k][c] for c in CONTRAST_COLS if c in ref[k] and c in v}} if k in ref else v) for k, v in tum.items()}; sens_res["host1_scouting_contrasts"], vv = run(swap, sc, base, bs, full=False)
        table["host1_scouting_contrasts"] = {"verdicts": vv, "changed": {k: {"primary": pv[k], "sensitivity": vv.get(k)} for k in pv if vv.get(k) != pv[k]}, "tumours_with_a_host1_contrast": sum(1 for k in tum if k in ref)}
    for name, ch in sens.items():
        r, vv = run(tum, sc, {**base, **ch}, bs, full=False); sens_res[name] = r; table[name] = {"verdicts": vv, "changed": {k: {"primary": pv[k], "sensitivity": vv.get(k)} for k in pv if vv.get(k) != pv[k]}}
    json.dump({"primary_verdicts": pv, "sensitivities": table}, open(os.path.join(out, "sensitivity_table.json"), "w"), indent=1, default=float); json.dump(sens_res, open(os.path.join(out, "sensitivities.json"), "w"), indent=1, default=float)
    L = ["# W4 analysis (PREREG.md applied as written)", "", "## Primary verdicts", ""] + [f"- {k}: **{v}**" for k, v in pv.items()] + ["", "## Sensitivities: verdicts that change against the primary", ""]
    for name, t in table.items():
        L.append(f"- {name}: " + ("no verdict changes" if not t["changed"] else "; ".join(f"{k}: {c['primary']} -> {c['sensitivity']}" for k, c in t["changed"].items())))
    for a, r in primary["b"].items():
        if r.get("status", "").startswith("OK"):
            c = r["coef"]["sign"]; L.append(f"- (b) {a}: sign OR {c['OR']:.2f} [{c['boot_OR_lo']:.2f}, {c['boot_OR_hi']:.2f}] (Wald [{c['wald_OR_lo']:.2f}, {c['wald_OR_hi']:.2f}]), discordant {r['discordant']} (non-flip {r['discordant_non_flip']}, A+P- {r['discordant_flip_A_bright_P_dark']}, A-P+ {r['discordant_flip_A_dark_P_bright']})")
    open(os.path.join(out, "ANALYSIS.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
