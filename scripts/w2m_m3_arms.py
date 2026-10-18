"""W2-M M3 arm selection (results/w2_mech/PREREG.md), from the UNEDITED W2 v3 kidney arm (kidney_133_1) and the ground-truth table; written to results/w2_mech/m3_arms.csv BEFORE any M3 run.
Arm A (dark to bright): main-set dark lesions (contrast_hu < 0) whose unedited run is not LEAK. Edit target L + |c|; sign-keeping control target T - 2|c|.
Arm B (bright to dark): main-set bright lesions (contrast_hu > 0) whose unedited run is LEAK. Edit target L - |c|; sign-keeping control target T + 2|c|. c = T - L (T lesion mean HU, L kidney-parenchyma 5-20 mm shell mean).
LEAK = predicted/lesion volume >= 3x or inside-lesion share <= 0.3. Usage: python scripts/w2m_m3_arms.py"""
import csv

pilot = {(r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv")) if r["organ"] == "kidney"}
gt = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2}
v3 = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/v3/w2v2_main_kidney.csv")) if r["arm"] == "kidney_133_1"}
rows = []
for k in sorted(v3):
    assert k in gt and k not in pilot
    r, g = v3[k], gt[k]; c = float(g["contrast_hu"]); leak = float(r["pred_over_lesion_volume"]) >= 3.0 or float(r["pred_frac_in_lesion"]) <= 0.3
    T, L = float(g["tumour_hu"]), float(g["liver_hu"]); arm = "A" if (c < 0 and not leak) else ("B" if (c > 0 and leak) else None)
    if arm:
        a = abs(T - L); edit, ctrl = (L + a, T - 2 * a) if arm == "A" else (L - a, T + 2 * a)
        rows.append({"arm": arm, "case": k[0], "rank": k[1], "diam_mm": round(float(g["diam_mm"]), 3), "contrast_hu": round(c, 2), "T": round(T, 3), "L": round(L, 3), "edit_target": round(edit, 3), "control_target": round(ctrl, 3), "baseline_dice_full": r["dice_full"]})
with open("results/w2_mech/m3_arms.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("arm A:", sum(r["arm"] == "A" for r in rows), "lesions; arm B:", sum(r["arm"] == "B" for r in rows), "lesions; cases:", len({r["case"] for r in rows}))
