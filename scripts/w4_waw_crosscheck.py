"""W4 scouting (CPU): cross-check of two runs of scripts/w4_waw_paired_contrast.py (the instance run that is used for the cohort, against a Mac re-run of a fixed random sample of patients).
Reports, over the patients present in both: the maximum absolute difference in every contrast column (HU), in the registration QA metrics (liver Dice, propagated tumour inside the target liver, volume ratio, centroid shift), and every change in QA pass/fail,
exclusion category (E, F, Q, K), bright/dark sign (contrast > 0 on each phase) and transform used. Optionally also the first Mac run's QA log lines (3-decimal values, default ITK threads) against the instance run's QA metrics.
Usage: python scripts/w4_waw_crosscheck.py <reference dir (instance run)> <comparison dir (Mac re-run)> <out json> [<first Mac run log with 'k / 199 patient dice .. inside .. ratio .. pass ..' lines>]"""
import csv
import json
import re
import sys


def flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(d):
    q = {r["patient"]: r for r in csv.DictReader(open(f"{d}/patient_qa.csv"))}; t = {(r["patient"], r["rank"]): r for r in csv.DictReader(open(f"{d}/tumours_paired.csv"))}; return q, t


def maxdiff(pairs):
    d = [abs(a - b) for a, b in pairs if a is not None and b is not None]; return max(d) if d else None


def main(ref, cmp, out, first_log=None):
    qr, tr = load(ref); qc, tc = load(cmp); pats = sorted(set(qr) & set(qc), key=int); keys = sorted((k for k in tr if k[0] in set(pats) and k in tc), key=lambda k: (int(k[0]), int(k[1])))
    res = {"patients_compared": len(pats), "patient_ids": pats, "tumours_compared": len(keys), "tumours_only_in_one_run": len([k for k in set(tr) ^ set(tc) if k[0] in set(pats)])}
    cols = ["c_drawn", "c_other", "c_arterial", "c_portal", "c_drawn_eroded", "c_other_eroded", "c_drawn_median", "c_other_median"]
    res["max_abs_contrast_difference_HU"] = {c: maxdiff([(flt(tr[k][c]), flt(tc[k][c])) for k in keys]) for c in cols}
    for c in ("c_drawn", "c_other"):
        d = sorted(((abs(flt(tr[k][c]) - flt(tc[k][c])), k) for k in keys if flt(tr[k][c]) is not None and flt(tc[k][c]) is not None), reverse=True); res[f"largest_{c}_differences"] = [{"patient": k[0], "rank": k[1], "abs_diff_HU": round(v, 3)} for v, k in d[:3]]
    qcols = ["liver_dice_none", "liver_dice_rigid", "liver_dice_final", "propagated_inside_target_liver", "volume_ratio", "tumour_centroid_shift_mm", "drawn_inside_own_liver"]
    res["max_abs_qa_difference"] = {c: maxdiff([(flt(qr[p][c]), flt(qc[p][c])) for p in pats]) for c in qcols}
    res["qa_pass_changes"] = [p for p in pats if qr[p]["qa_pass"] != qc[p]["qa_pass"]]; res["transform_changes"] = [p for p in pats if qr[p]["transform"] != qc[p]["transform"]]
    res["category_changes"] = [{"patient": k[0], "rank": k[1], "ref": tr[k]["category"], "cmp": tc[k]["category"]} for k in keys if tr[k]["category"] != tc[k]["category"]]
    sg = lambda r, c: (None if flt(r[c]) is None else flt(r[c]) > 0)      # noqa: E731
    res["bright_dark_sign_changes"] = [{"patient": k[0], "rank": k[1], "column": c} for k in keys for c in ("c_arterial", "c_portal") if sg(tr[k], c) != sg(tc[k], c)]
    if first_log:
        rows = {}
        for line in open(first_log):
            m = re.match(r"\d+ / \d+ (\d+) dice ([\d.]+) inside ([\d.na]+) ratio ([\d.na]+) pass (\d)", line)
            if m:
                rows[m.group(1)] = (float(m.group(2)), flt(m.group(3)), flt(m.group(4)), int(m.group(5)))
        both = [p for p in rows if p in qr]; res["first_mac_run_log_vs_instance"] = {"patients": len(both), "max_abs_dice_diff": maxdiff([(rows[p][0], flt(qr[p]["liver_dice_final"])) for p in both]), "max_abs_inside_diff": maxdiff([(rows[p][1], flt(qr[p]["propagated_inside_target_liver"])) for p in both]),
                                      "max_abs_ratio_diff": maxdiff([(rows[p][2], flt(qr[p]["volume_ratio"])) for p in both]), "qa_pass_changes": [p for p in both if rows[p][3] != int(qr[p]["qa_pass"])],
                                      "note": "the first Mac run used the default ITK thread count (all cores per worker, 3 workers) and the same code; its log values are rounded to 3 decimals"}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
