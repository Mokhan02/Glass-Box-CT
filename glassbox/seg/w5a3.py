"""W5 Amendment 3 statistics (CPU; results/w5_training/PREREG.md, Amendment 3): the paired M3 contrast between BRIGHT-50 and D0.

Per lesion D_A = delta_cov(edit) - delta_cov(control A) (delta_cov = edited minus baseline coverage), read from the M3 runner's m3_lesions.csv
(base / edit / control rows per lesion). Lesions are matched by (case, rank). The edited volumes depend only on the scan and the ground truth, so both
models were given the same edits. Delta_A = mean over the lesions eligible in BOTH models of [D_A(BRIGHT-50) - D_A(D0)] (positive = less harm in BRIGHT-50),
case-level bootstrap (1000 resamples, seed = first 8 hex digits of SHA-256 of the name `w5a3_m3_contrast_A`). Verdict (margin 0.10, fixed before any run), in this order:
REDUCED if the CI lower bound > 0 and the point estimate >= +0.10; NOT REDUCED if the CI upper bound < +0.10; INCONCLUSIVE otherwise; fewer than 10 common lesions: INCONCLUSIVE."""
from __future__ import annotations

import csv

import numpy as np

from glassbox.seg import w3s2_stats as S

MARGIN = 0.10
MIN_LESIONS = 10
SEED_NAME = "w5a3_m3_contrast_A"
B = S.B_DEFAULT


def load_m3(path, cases=None):
    """{(case, rank): {condition: row}} from an M3 runner CSV, restricted to `cases` when given."""
    by = {}
    for r in csv.DictReader(open(path)):
        if cases is not None and r["case"] not in cases:
            continue
        by.setdefault((r["case"], str(r["rank"])), {})[r["condition"]] = r
    return by


def per_lesion(by, arm="A"):
    """{key: dict} for lesions with base, edit and control rows whose arm is `arm`: delta_cov(edit), delta_cov(control), D (coverage), D (Dice within 10 mm), flagged."""
    out = {}
    for k, v in by.items():
        if not all(c in v for c in ("base", "edit", "control")) or v["edit"]["arm"] != arm:
            continue
        f = lambda cond, m: float(v[cond][m]) - float(v["base"][m])      # noqa: E731
        out[k] = {"case": k[0], "dcov_edit": f("edit", "coverage"), "dcov_ctrl": f("control", "coverage"), "D": f("edit", "coverage") - f("control", "coverage"),
                  "D_dice": f("edit", "dice_within_10mm") - f("control", "dice_within_10mm"), "dice_edit": f("edit", "dice_within_10mm"),
                  "flagged": str(v["edit"].get("flagged")) == "True" or str(v["control"].get("flagged")) == "True", "base_cov": float(v["base"]["coverage"])}
    return out


def verdict(point, lo, hi, n, margin=MARGIN):
    if n < MIN_LESIONS:
        return "INCONCLUSIVE"
    if lo > 0 and point >= margin:
        return "REDUCED"
    if hi < margin:
        return "NOT REDUCED"
    return "INCONCLUSIVE"


def _paired(a, b, keys, field, name):
    """Mean over `keys` of (b - a) with the case-level bootstrap: (point, lo, hi)."""
    d = np.array([b[k][field] - a[k][field] for k in keys]); cs = [k[0] for k in keys]
    return S.boot_stat(cs, d, np.mean, name, B)


def contrast(d0, b50, arm="A"):
    """The primary contrast and its descriptive companions. `d0` and `b50` are load_m3 results (already restricted to the cohort cases)."""
    a0, a1 = per_lesion(d0, arm), per_lesion(b50, arm); common = sorted(set(a0) & set(a1)); n = len(common)
    res = {"arm": arm, "n_eligible_d0": len(a0), "n_eligible_bright50": len(a1), "n_common": n, "n_cases_common": len({k[0] for k in common}),
           "d0_only": sorted(set(a0) - set(a1)), "bright50_only_count": len(set(a1) - set(a0)), "margin": MARGIN, "seed_name": SEED_NAME}
    if n < 2:
        res.update({"delta": None, "verdict": "INCONCLUSIVE"}); return res
    point, lo, hi = _paired(a0, a1, common, "D", SEED_NAME)
    mean_d0 = float(np.mean([a0[k]["D"] for k in common])); mean_b = float(np.mean([a1[k]["D"] for k in common]))
    res.update({"delta": {"point": point, "ci": [lo, hi]}, "verdict": verdict(point, lo, hi, n), "mean_D_d0_common": mean_d0, "mean_D_bright50_common": mean_b,
                "relative_reduction_vs_d0_harm": (point / abs(mean_d0)) if mean_d0 else None})
    # descriptive companions (no verdict)
    res["descriptive"] = {
        "delta_cov_edit_alone": dict(zip(("point", "lo", "hi"), _paired({k: {"x": a0[k]["dcov_edit"]} for k in common}, {k: {"x": a1[k]["dcov_edit"]} for k in common}, common, "x", "w5a3_m3_edit_alone"))),
        "lesion_dice_D": dict(zip(("point", "lo", "hi"), _paired({k: {"x": a0[k]["D_dice"]} for k in common}, {k: {"x": a1[k]["D_dice"]} for k in common}, common, "x", "w5a3_m3_dice_D"))),
        "median_difference_common": float(np.median([a1[k]["D"] - a0[k]["D"] for k in common])),
    }
    nf = [k for k in common if not (a0[k]["flagged"] or a1[k]["flagged"])]
    if len(nf) >= 2:
        p2, l2, h2 = _paired(a0, a1, nf, "D", SEED_NAME + "_nf")
        res["sensitivity_excluding_flagged"] = {"n": len(nf), "point": p2, "ci": [l2, h2], "verdict": verdict(p2, l2, h2, len(nf))}
    # each model on its own eligible set (D0's M3 statistics), residual harm in BRIGHT-50
    for name, a in (("d0_own", a0), ("bright50_own", a1)):
        ks = sorted(a); cs = [k[0] for k in ks]
        if len(ks) >= 2:
            st = S.m3_arm_stats(cs, np.array([a[k]["dcov_edit"] for k in ks]), np.array([a[k]["D"] for k in ks]), f"w5a3_{name}_{arm}", -1 if arm == "A" else 1)
            st["residual_harm"] = ("PRESENT" if st["median_D"][2] < 0 and st["sign_test"]["p"] < 0.05 else "ABSENT") if arm == "A" else None
            res[name] = st
    # unpaired difference of medians on the own sets (case bootstrap over the union of cases)
    if len(a0) >= 2 and len(a1) >= 2:
        cases = sorted({k[0] for k in a0} | {k[0] for k in a1}); rng = np.random.default_rng(S.arm_seed("w5a3_m3_unpaired_A")); idx0 = {c: [a0[k]["D"] for k in a0 if k[0] == c] for c in cases}; idx1 = {c: [a1[k]["D"] for k in a1 if k[0] == c] for c in cases}
        draws = []
        for row in rng.integers(0, len(cases), (B, len(cases))):
            x0 = [v for i in row for v in idx0[cases[i]]]; x1 = [v for i in row for v in idx1[cases[i]]]
            if x0 and x1:
                draws.append(np.median(x1) - np.median(x0))
        res["unpaired_median_difference"] = {"point": float(np.median([a1[k]["D"] for k in a1]) - np.median([a0[k]["D"] for k in a0])), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return res


def arm_b_listing(d0, b50):
    """Arm B (bright -> dark) is below the 10-lesion minimum for BRIGHT-50: list each BRIGHT-50 Arm B lesion beside D0's values for the same lesion (descriptive, no verdict)."""
    b0, b1 = per_lesion(d0, "B"), per_lesion(b50, "B"); rows = []
    for k in sorted(b1):
        rows.append({"case": k[0], "rank": k[1], "bright50": {x: b1[k][x] for x in ("base_cov", "dcov_edit", "dcov_ctrl", "D")}, "d0": ({x: b0[k][x] for x in ("base_cov", "dcov_edit", "dcov_ctrl", "D")} if k in b0 else None)})
    return {"n_bright50": len(b1), "n_d0": len(b0), "min_required": MIN_LESIONS, "verdict": "NONE (fewer than 10 lesions)" if len(b1) < MIN_LESIONS else "not pre-registered for contrast", "lesions": rows}


def m2_by_group(by, group_of, conds=("leace", "random", "hu")):
    """EXPLORATORY (Amendment 3, no verdict): the M2 lesion-Dice change under each eraser, separately for bright and dark lesions the unhooked model detects.
    `by` = {(case, rank): {cond: row}} from the M2 runner's lesions.csv (cond = base / leace / random / hu); `group_of` = {(case, rank): "bright" | "dark"}."""
    out = {}
    for g in ("bright", "dark"):
        ks = sorted(k for k, v in by.items() if group_of.get(k) == g and all(c in v for c in ("base",) + tuple(conds)) and float(v["base"]["coverage"]) >= 0.10)
        res = {"n_lesions": len(ks), "n_cases": len({k[0] for k in ks})}
        if len(ks) >= 2:
            cs = [k[0] for k in ks]
            for c in conds:
                d = np.array([float(by[k][c]["dice_within_10mm"]) - float(by[k]["base"]["dice_within_10mm"]) for k in ks])
                res[c] = {"mean_dice_change": dict(zip(("point", "lo", "hi"), S.boot_stat(cs, d, np.mean, f"w5a3_m2_{g}_{c}", B))), "still_detected": float(np.mean([float(by[k][c]["coverage"]) >= 0.10 for k in ks]))}
        out[g] = res
    return out
