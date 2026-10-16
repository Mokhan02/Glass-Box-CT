"""W1 pre-registered comparison (results/w1_training_audit/PREREG.md). Pooled primary first; per-dataset, size strata and sensitivities are SECONDARY.
Inputs: crml_lesions.csv, msd_lesions.csv (training sets), results/exp8_lesions/lesion_table.csv route==uid (HCC reference, 86 lesions / 75 cases).
Primary unit: the 2 largest lesions >= 10 mm per case (rank <= 2). Bright = contrast > 0 HU. Case-level bootstrap, 1000 resamples, seed 0, percentile 95% CIs.
d = p_HCC - p_train; margin 0.10; training side needs >= 200 lesions. Verdict order: BRIGHT RARE IN TRAINING, REVERSED, NO GAP, INCONCLUSIVE.
Usage: python scripts/w1_compare.py results/w1_training_audit"""
import csv
import json
import sys

import numpy as np

out = sys.argv[1]
MARGIN, N_MIN, B, SEED = 0.10, 200, 1000, 0
rd = lambda p: list(csv.DictReader(open(p)))
crml, msd = rd(f"{out}/crml_lesions.csv"), rd(f"{out}/msd_lesions.csv")
hcc = [r for r in rd("results/exp8_lesions/lesion_table.csv") if r["route"] == "uid"]


def rows(src, key, col="contrast_hu"):
    return [{"case": r[key], "rank": int(r["rank"]), "diam": float(r["diam_mm"]), "c": float(r[col]) if r[col] not in ("", "nan") else np.nan, "dataset": r.get("dataset", "hcc"), "included_dev4": r.get("included_dev4", "True") == "True"} for r in src]


def boot_share(rs, pred=lambda r: r["c"] > 0, rng=None):
    """Case-level bootstrap of the share of lesions satisfying pred. Returns point, lo, hi, n_lesions, n_cases, and the bootstrap draws."""
    rs = [r for r in rs if not np.isnan(r["c"])]
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}
    cnt = np.array([[sum(pred(r) for r in by[c]), len(by[c])] for c in cases])
    point = cnt[:, 0].sum() / cnt[:, 1].sum(); idx = rng.integers(0, len(cases), (B, len(cases)))
    draws = cnt[idx, 0].sum(1) / cnt[idx, 1].sum(1)
    return {"share": point, "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)), "n_lesions": int(cnt[:, 1].sum()), "n_cases": len(cases), "bright": int(cnt[:, 0].sum())}, draws


def compare(train, ref, rng):
    t, dt = boot_share(train, rng=rng); h, dh = boot_share(ref, rng=rng); d = dh - dt                   # independent bootstraps of the two shares
    return {"p_train": t, "p_hcc": h, "d": h["share"] - t["share"], "d_lo": float(np.percentile(d, 2.5)), "d_hi": float(np.percentile(d, 97.5))}


def verdict(r):
    if r["p_train"]["n_lesions"] < N_MIN:
        return "INCONCLUSIVE (training side has fewer than 200 lesions)"
    if r["d_lo"] > 0 and r["d"] >= MARGIN:
        return "BRIGHT RARE IN TRAINING"
    if r["d_hi"] < 0:
        return "REVERSED"
    if r["d_lo"] >= -MARGIN and r["d_hi"] <= MARGIN:
        return "NO GAP"
    return "INCONCLUSIVE"


prim = lambda rs: [r for r in rs if r["rank"] <= 2]
H = prim(rows(hcc, "case")); C = prim(rows(crml, "patient")); M = prim(rows(msd, "patient"))
res = {"primary": {}, "secondary": {}}
rng = np.random.default_rng(SEED)
res["primary"] = compare(C + M, H, rng); res["primary"]["verdict"] = verdict(res["primary"])
S = res["secondary"]
S["per_dataset"] = {n: (lambda r: {**r, "verdict": verdict(r)})(compare(t, H, rng)) for n, t in (("msd_task03", M), ("crml_ct", C))}
q = lambda rs: {f"p{k}": float(np.nanpercentile([r["c"] for r in rs], k)) for k in (5, 25, 50, 75, 95)}
S["contrast_quantiles"] = {"msd_task03": q(M), "crml_ct": q(C), "pooled_training": q(C + M), "hcc_uid": q(H)}
bins = (("<30 mm", lambda r: r["diam"] < 30), ("30-60 mm", lambda r: 30 <= r["diam"] <= 60), (">60 mm", lambda r: r["diam"] > 60))
S["size_strata"] = {}
for name, f in bins:
    tr, hh = [r for r in C + M if f(r)], [r for r in H if f(r)]
    S["size_strata"][name] = compare(tr, hh, rng) if tr and hh else {"note": "empty stratum", "n_train": len(tr), "n_hcc": len(hh)}
sens = S["sensitivity"] = {}
def comp_pred(train, ref, pred):
    t, dt = boot_share(train, pred=pred, rng=rng); h, dh = boot_share(ref, pred=pred, rng=rng); d = dh - dt
    return {"p_train": t, "p_hcc": h, "d": h["share"] - t["share"], "d_lo": float(np.percentile(d, 2.5)), "d_hi": float(np.percentile(d, 97.5))}
sens["bright_gt_+10HU"] = comp_pred(C + M, H, lambda r: r["c"] > 10)
sens["crml_no_hepatic_vein_reference"] = compare(prim(rows(crml, "patient", "contrast_hu_no_hepatic_vein")) + M, H, rng)
sens["crml_no_vessel_removal_reference"] = compare(prim(rows(crml, "patient", "contrast_hu_no_vessel_removal")) + M, H, rng)
M4 = [r for r in M if r["included_dev4"]]
sens["msd_inclusion_as_in_DEV4_(voxel-fraction_guard)"] = compare(C + M4, H, rng)
sens["all_lesions_ge_10mm_training_vs_hcc_primary"] = compare(rows(crml, "patient") + rows(msd, "patient"), H, rng)
import os
if os.path.exists(f"{out}/msd_lesions_guard_failed.csv"):      # appended LAST so no earlier draw changes
    GF = prim(rows(rd(f"{out}/msd_lesions_guard_failed.csv"), "patient"))
    sens["POST HOC (added after the primary result was known; signed off by the researcher): MSD cases that failed the DEV5 surface guard included"] = compare(C + M + GF, H, rng)
json.dump(res, open(f"{out}/w1_results.json", "w"), indent=1, default=float)
f = lambda s: f"{100 * s['share']:.1f}% [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}] ({s['bright']}/{s['n_lesions']} lesions, {s['n_cases']} cases)"
g = lambda r: f"d = {100 * r['d']:+.1f} pts [{100 * r['d_lo']:+.1f}, {100 * r['d_hi']:+.1f}]"
L = ["# W1 results (generated by scripts/w1_compare.py; the verdict is the pre-registered rule applied as written)", "",
     "## Primary: pooled training data (MSD Task03 + CRML-CT) vs HCC UID-route, 2 largest lesions >= 10 mm per case",
     f"- p_train (bright share, training): {f(res['primary']['p_train'])}", f"- p_HCC (held-out, UID route): {f(res['primary']['p_hcc'])}",
     f"- {g(res['primary'])}; margin 0.10; training lesions {res['primary']['p_train']['n_lesions']} (needs >= {N_MIN})", f"- **Verdict: {res['primary']['verdict']}**", "",
     "## Secondary (does not change the verdict)", "### Per dataset"]
for n, r in S["per_dataset"].items():
    L.append(f"- {n}: p_train {f(r['p_train'])}; {g(r)}; rule applied alone: {r['verdict']}")
L += ["### Contrast quantiles (HU, 5/25/50/75/95)"] + [f"- {n}: " + ", ".join(f"{v:+.1f}" for v in qq.values()) for n, qq in S["contrast_quantiles"].items()] + ["### Size strata (pooled training vs HCC)"]
for n, r in S["size_strata"].items():
    L.append(f"- {n}: " + (f"p_train {f(r['p_train'])}; p_HCC {f(r['p_hcc'])}; {g(r)}" if "p_train" in r else f"{r['note']} (train {r['n_train']}, HCC {r['n_hcc']})"))
L += ["### Sensitivity analyses"] + [f"- {n}: p_train {f(r['p_train'])}; p_HCC {f(r['p_hcc'])}; {g(r)}" for n, r in sens.items() if r]
open(f"{out}/w1_results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
