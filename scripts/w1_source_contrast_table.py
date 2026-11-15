"""[EXPLORATORY / DESCRIPTIVE, requested by the researcher 2026-11-15; no pass/fail rule] Bright versus dark tumours (and the contrast distribution) in VISTA-3D's liver-tumour training sources and in the two held-out reference cohorts, under the frozen W1 contrast definition (results/w1_training_audit/PREREG.md: c = tumour mean HU minus the mean HU of non-tumour liver in the 5-20 mm shell, 200-voxel fallback; bright = c > 0 HU; lesions >= 10 mm; PRIMARY UNIT = the 2 largest lesions per case; all lesions reported as a second line).
Inputs are the per-lesion tables already produced by the frozen pipeline (no contrast is recomputed here): results/w1_training_audit/{msd_lesions,crml_lesions,hcc_lesions_variants}.csv, results/w4_waw/contrast_masked_phase.csv and results/w4_waw/paired_instances/tumours_paired.csv, plus an optional Task08 table (--task08 <csv>, same columns as msd_lesions.csv).
Bright-share CIs: case-level bootstrap (1000 resamples, seed 0, percentile 95%). Usage: python scripts/w1_source_contrast_table.py <out dir> [--task08 <csv>]"""
import csv
import json
import os
import sys

import numpy as np

R = "results"


def load(path, case, c, rank=None, extra=None):
    rows = []
    for r in csv.DictReader(open(path)):
        d = {"case": r[case], "c": float(r[c]), "rank": int(r[rank]) if rank else 0}
        if extra:
            d.update({k: r[v] for k, v in extra.items()})
        rows.append(d)
    return rows


def share(rows, B=1000):
    if not rows:
        return {"n_lesions": 0, "n_cases": 0, "bright": 0, "dark": 0, "pct_bright": None, "lo": None, "hi": None, "q": None}
    cases = sorted({r["case"] for r in rows}); by = {k: [r["c"] > 0 for r in rows if r["case"] == k] for k in cases}; rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        pick = rng.choice(len(cases), len(cases)); v = [x for i in pick for x in by[cases[i]]]; bs.append(np.mean(v))
    c = np.array([r["c"] for r in rows]); b = int((c > 0).sum())
    return {"n_lesions": len(rows), "n_cases": len(cases), "bright": b, "dark": len(rows) - b, "pct_bright": 100 * b / len(rows), "lo": 100 * float(np.percentile(bs, 2.5)), "hi": 100 * float(np.percentile(bs, 97.5)), "q": [float(x) for x in np.percentile(c, [5, 25, 50, 75, 95])]}


def main(argv):
    out = argv[0]; os.makedirs(out, exist_ok=True); t8 = argv[argv.index("--task08") + 1] if "--task08" in argv else None
    sets = {}      # name -> (phase note, primary rows, all rows)
    m = load(f"{R}/w1_training_audit/msd_lesions.csv", "patient", "contrast_hu", "rank"); sets["MSD Task03 (training source)"] = ("portal-venous in most cases (LiTS, several institutions); phase not labelled per case", [r for r in m if r["rank"] <= 2], m)
    if t8:
        m8 = load(t8, "patient", "contrast_hu", "rank"); sets["MSD Task08 HepaticVessel (training source; PSEUDO-LIVER)"] = ("portal-venous per the MSD description (not verified per case); liver mask = TotalSegmentator pseudo-liver", [r for r in m8 if r["rank"] <= 2], m8)
    c = load(f"{R}/w1_training_audit/crml_lesions.csv", "patient", "contrast_hu", "rank"); sets["CRML-CT (training source)"] = ("portal-venous (preoperative contrast-enhanced CT)", [r for r in c if r["rank"] <= 2], c)
    h = load(f"{R}/w1_training_audit/hcc_lesions_variants.csv", "case", "contrast_hu", "rank"); sets["HCC-TACE-Seg (held-out reference)"] = ("mixed acquisitions; phase not labelled (tags stripped)", h, None)
    w = [r for r in csv.DictReader(open(f"{R}/w4_waw/contrast_masked_phase.csv"))]; wr = [{"case": r["patient"], "c": float(r["contrast_hu"]), "ph": r["phase"], "top2": r["top2"] == "1"} for r in w]
    sets["WAW-TACE, contrast on the phase the mask was drawn on (held-out reference)"] = ("arterial (phase 1) or portal-venous (phase 2), whichever the annotators drew on; 181 of 276 primary lesions drawn on arterial", [r for r in wr if r["top2"]], wr)
    p = [r for r in csv.DictReader(open(f"{R}/w4_waw/paired_instances/tumours_paired.csv")) if r["category"] == "K"]
    sets["WAW-TACE arterial series, all 297 paired tumours"] = ("arterial", [{"case": r["patient"], "c": float(r["c_arterial"])} for r in p], None)
    sets["WAW-TACE portal-venous series, same 297 tumours"] = ("portal-venous", [{"case": r["patient"], "c": float(r["c_portal"])} for r in p], None)
    res, md = {}, ["| Dataset | Phase | Cases | Lesions (2 largest per case) | Bright (c > 0) | Dark (c <= 0) | Bright share [95% CI] | c quantiles 5/25/50/75/95 (HU) | All lesions >= 10 mm: bright / total |", "|---|---|---|---|---|---|---|---|---|"]
    for name, (ph, prim, allr) in sets.items():
        s = share(prim); a = share(allr) if allr else None; res[name] = {"phase": ph, "primary": s, "all_lesions": a}
        md.append(f"| {name} | {ph} | {s['n_cases']} | {s['n_lesions']} | {s['bright']} | {s['dark']} | {s['pct_bright']:.1f}% [{s['lo']:.1f}, {s['hi']:.1f}] | {', '.join(f'{x:+.0f}' for x in s['q'])} | " + (f"{a['bright']} / {a['n_lesions']}" if a else "n/a (primary unit only)") + " |")
    ws = {}
    for ph, nm in (("1", "arterial-drawn"), ("2", "portal-drawn")):
        ws[nm] = share([r for r in wr if r["top2"] and r["ph"] == ph])
    res["waw_by_drawn_phase"] = ws
    md += ["", "WAW-TACE primary lesions by the phase the mask was drawn on: " + "; ".join(f"{k}: {v['bright']}/{v['n_lesions']} bright ({v['pct_bright']:.1f}%)" for k, v in ws.items())]
    open(os.path.join(out, "source_contrast_table.md"), "w").write("\n".join(md) + "\n"); json.dump(res, open(os.path.join(out, "source_contrast_table.json"), "w"), indent=1)
    with open(os.path.join(out, "source_contrast_lesions.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["dataset", "case", "contrast_hu", "primary_unit"])
        for name, (ph, prim, allr) in sets.items():
            ids = {id(x) for x in prim}
            for r in (allr if allr else prim):
                wtr.writerow([name, r["case"], f"{r['c']:.3f}", int(id(r) in ids)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [(n, s[1]) for n, s in sets.items() if "series" not in n]; fig, ax = plt.subplots(len(show), 1, figsize=(8, 1.9 * len(show)), sharex=True); bins = np.arange(-120, 125, 5)
    for a, (n, rows) in zip(np.atleast_1d(ax), show):
        c = np.clip([r["c"] for r in rows], -119.9, 119.9); a.hist(c[c <= 0], bins=bins, color="#3b6ea5"); a.hist(c[c > 0], bins=bins, color="#d9822b"); a.axvline(0, color="k", lw=1)
        sh = share(rows); a.set_ylabel("lesions", fontsize=8); a.set_title(f"{n}: n {sh['n_lesions']}, bright {sh['bright']} ({sh['pct_bright']:.1f}%)", fontsize=8, loc="left"); a.tick_params(labelsize=8)
    np.atleast_1d(ax)[-1].set_xlabel("tumour minus liver contrast c (HU); values beyond +-120 HU are in the edge bins; blue = dark (c <= 0), orange = bright (c > 0)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "source_contrast_histogram.png"), dpi=150)
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
