"""W4 scouting (CPU): exclusion categories (E, F, Q, K), sign-flip counts and the effective N for the sign-versus-magnitude test, exactly as fixed in results/w4_waw/CONTRAST_RULES.md BEFORE the cohort run.
Inputs: the paired run's tumours_paired.csv and patient_qa.csv, and the masked-phase contrast csv (the universe of tumours >= 10 mm on the drawn phase, independent of registration), optionally the cohort mask check json (patients with tumour files holding the value 2).
Definitions (fixed): sign flip = c_arterial and c_portal have opposite signs AND both |c| >= delta; comparable |contrast| = both |c| >= delta AND min/max of the two |c| >= rho; effective N = flips that are also comparable.
Primary delta = 5 HU, rho = 0.5; the full grid delta {0, 5, 10} x rho {0.33, 0.5, 0.67} is reported, nothing is chosen from it. No detection outcome is read or computed here.
Usage: python scripts/w4_waw_sign_counts.py <paired dir> <masked-phase contrast csv> <out dir> [<mask_check_cohort.json>]"""
import csv
import json
import os
import sys
from collections import Counter

DELTAS, RHOS, PRIMARY = (0.0, 5.0, 10.0), (0.33, 0.5, 0.67), (5.0, 0.5)


def flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flip(a, p, delta):
    return a is not None and p is not None and a * p < 0 and abs(a) >= delta and abs(p) >= delta


def comparable(a, p, delta, rho):
    if a is None or p is None or abs(a) < delta or abs(p) < delta:
        return False
    hi = max(abs(a), abs(p))
    return hi > 0 and min(abs(a), abs(p)) / hi >= rho


def counts(rows, col_a="c_arterial", col_p="c_portal"):
    """The grid of counts over the given tumour rows (each a dict with the two contrast columns)."""
    out = {}
    for d in DELTAS:
        for r in RHOS:
            fl = cm = both = 0
            for x in rows:
                a, p = flt(x[col_a]), flt(x[col_p])
                f, c = flip(a, p, d), comparable(a, p, d, r); fl += f; cm += c; both += (f and c)
            out[f"delta={d:g},rho={r:g}"] = {"sign_flips": fl, "comparable_abs_contrast": cm, "flip_and_comparable (effective N)": both}
    return out


def concordance(rows, delta=0.0):
    """2 x 2 of the contrast signs (arterial, portal) among tumours with both contrasts; |c| >= delta in both."""
    t = Counter()
    for x in rows:
        a, p = flt(x["c_arterial"]), flt(x["c_portal"])
        if a is None or p is None or abs(a) < delta or abs(p) < delta:
            t["excluded_by_dead_zone"] += 1; continue
        t[("A+" if a > 0 else "A-") + "/" + ("P+" if p > 0 else "P-")] += 1
    return dict(t)


def main(pdir, masked_csv, out, mask_check=None):
    os.makedirs(out, exist_ok=True)
    tum = list(csv.DictReader(open(os.path.join(pdir, "tumours_paired.csv")))); qa = {r["patient"]: r for r in csv.DictReader(open(os.path.join(pdir, "patient_qa.csv")))}
    universe = [(r["patient"], r["phase"], r["rank"]) for r in csv.DictReader(open(masked_csv))]; have = {(r["patient"], str(r["mask_phase"]), r["rank"]) for r in tum}
    cat = Counter(r["category"] for r in tum)
    e_pat = sorted(p for p, r in qa.items() if r.get("error")); missing = [u for u in universe if u not in have]; cat["E"] = len(missing)      # tumours of patients whose run failed (no rows)
    res = {"universe_tumours_ge10mm_on_drawn_phase": len(universe), "tumours_in_paired_csv": len(tum), "patients": len(qa), "patients_E_error": e_pat, "categories": dict(cat), "category_check_sum": sum(cat.values())}
    res["patients_qa_pass"] = sum(int(r["qa_pass"]) for r in qa.values()); res["patients_with_any_fov_clipped_tumour"] = sum(1 for r in qa.values() if int(r.get("n_lesions_fov_clipped") or 0) > 0)
    kept = [r for r in tum if r["category"] == "K"]; res["kept_by_drawn_phase"] = dict(Counter(r["mask_phase"] for r in kept))
    q_fail_reasons = Counter()
    for r in tum:
        if r["category"] == "Q":
            q = qa[r["patient"]]; d, i, v = flt(q["liver_dice_final"]), flt(q["propagated_inside_target_liver"]), flt(q["volume_ratio"])
            q_fail_reasons["liver_dice<0.92" if d is not None and d < 0.92 else ("inside<0.90" if i is not None and i < 0.90 else ("volume_ratio" if v is not None else "other"))] += 1
    res["Q_first_failed_criterion_per_tumour"] = dict(q_fail_reasons)
    res["contrast_sign_concordance_kept(delta=0)"] = concordance(kept, 0.0); res["contrast_sign_concordance_kept(delta=5)"] = concordance(kept, 5.0)
    res["bright_share_kept"] = {"arterial_contrast>0": round(sum(flt(r["c_arterial"]) > 0 for r in kept) / max(len(kept), 1), 4), "portal_contrast>0": round(sum(flt(r["c_portal"]) > 0 for r in kept) / max(len(kept), 1), 4)}
    res["grid_all_kept"] = counts(kept); res["by_drawn_phase"] = {ph: counts([r for r in kept if r["mask_phase"] == ph]) for ph in ("1", "2")}
    res["sensitivity_primary_setting"] = {"liver_eroded_3mm": counts(kept, "c_arterial_eroded", "c_portal_eroded")[f"delta={PRIMARY[0]:g},rho={PRIMARY[1]:g}"], "shell_median": counts(kept, "c_arterial_median", "c_portal_median")[f"delta={PRIMARY[0]:g},rho={PRIMARY[1]:g}"]}
    if mask_check and os.path.exists(mask_check):
        mc = json.load(open(mask_check))["patients"]; v2 = sorted({t["file"].split("_")[0] for e in mc.values() for t in e["tumour"] if not t["values_exactly_0_1"]}, key=int)
        sub = [r for r in kept if r["patient"] not in set(v2)]; res["excluding_patients_with_value_2_tumour_files"] = {"patients": v2, "kept_remaining": len(sub), "primary": counts(sub)[f"delta={PRIMARY[0]:g},rho={PRIMARY[1]:g}"]}
    json.dump(res, open(os.path.join(out, "sign_counts.json"), "w"), indent=1)
    p = res["grid_all_kept"][f"delta={PRIMARY[0]:g},rho={PRIMARY[1]:g}"]
    L = ["# W4 sign counts (rules fixed in CONTRAST_RULES.md before the cohort run; no detection outcome)", "", f"Universe: {len(universe)} tumours >= 10 mm on the drawn phase; categories (each tumour in exactly one): {dict(cat)}; sum {sum(cat.values())}.", "",
         f"Kept (K): {len(kept)} tumours; concordance of the contrast signs (arterial/portal, delta 0): {res['contrast_sign_concordance_kept(delta=0)']}.", "",
         f"**Primary setting (delta 5 HU, rho 0.5): sign flips {p['sign_flips']}, comparable |contrast| {p['comparable_abs_contrast']}, effective N (flip and comparable) {p['flip_and_comparable (effective N)']}.**", "", "| setting | sign flips | comparable | effective N |", "|---|---|---|---|"]
    L += [f"| {k} | {v['sign_flips']} | {v['comparable_abs_contrast']} | {v['flip_and_comparable (effective N)']} |" for k, v in res["grid_all_kept"].items()]
    open(os.path.join(out, "sign_counts.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
