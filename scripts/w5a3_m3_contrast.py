"""W5 Amendment 3, primary analysis (CPU): the paired M3 contrast BRIGHT-50 minus D0 (results/w5_training/PREREG.md, Amendment 3; statistics in glassbox/seg/w5a3.py).
Usage: python scripts/w5a3_m3_contrast.py <D0 m3_lesions.csv> <BRIGHT-50 m3_lesions.csv> <cases file> <out dir>
Writes m3_contrast.json and m3_contrast.md. The verdict (REDUCED / NOT REDUCED / INCONCLUSIVE) is the pre-registered rule applied as written."""
import json
import os
import sys

sys.path.insert(0, ".")
from glassbox.seg import w5a3 as W  # noqa: E402

d0p, b50p, cases_file, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
cases = {l.strip() for l in open(cases_file) if l.strip()}
d0, b50 = W.load_m3(d0p, cases), W.load_m3(b50p, cases)
res = {"contrast_arm_A": W.contrast(d0, b50, "A"), "arm_B": W.arm_b_listing(d0, b50), "cases": len(cases), "inputs": {"d0": d0p, "bright50": b50p}}
json.dump(res, open(os.path.join(out, "m3_contrast.json"), "w"), indent=1, default=float)
c = res["contrast_arm_A"]; f = lambda x: "n/a" if x is None else f"{x:+.3f}"      # noqa: E731
L = ["# W5 Amendment 3, M3 contrast: BRIGHT-50 minus D0 (Arm A, dark detected lesions edited to bright)", "",
     f"Common lesions (eligible in both models): **{c['n_common']}** in {c['n_cases_common']} cases (D0 eligible {c['n_eligible_d0']}, BRIGHT-50 eligible {c['n_eligible_bright50']}; D0 lesions not eligible in BRIGHT-50: {len(c['d0_only'])}).", ""]
if c.get("delta"):
    d = c["delta"]
    L += [f"**Delta_A = mean [D_A(BRIGHT-50) - D_A(D0)] = {f(d['point'])} [{f(d['ci'][0])}, {f(d['ci'][1])}]**; verdict (margin {W.MARGIN}): **{c['verdict']}**.",
          f"D0 mean D_A on the common set {f(c['mean_D_d0_common'])}; BRIGHT-50 {f(c['mean_D_bright50_common'])}; relative reduction {c['relative_reduction_vs_d0_harm']:.0%} (descriptive).", ""]
    for k in ("d0_own", "bright50_own"):
        if k in c:
            s = c[k]; L.append(f"- {k}: n {s['n']} lesions / {s['n_cases']} cases; median delta_cov(edit) {f(s['median_edit'][0])} [{f(s['median_edit'][1])}, {f(s['median_edit'][2])}]; median D_A {f(s['median_D'][0])} [{f(s['median_D'][1])}, {f(s['median_D'][2])}]; sign test {s['sign_test']['k']}/{s['sign_test']['n']} p = {s['sign_test']['p']:.2g}" + (f"; residual Arm A harm {s['residual_harm']}" if s.get("residual_harm") else ""))
    ds = c["descriptive"]; L += ["", "Descriptive companions (no verdict): delta_cov(edit) alone " + f"{f(ds['delta_cov_edit_alone']['point'])} [{f(ds['delta_cov_edit_alone']['lo'])}, {f(ds['delta_cov_edit_alone']['hi'])}]; lesion Dice D_A " + f"{f(ds['lesion_dice_D']['point'])} [{f(ds['lesion_dice_D']['lo'])}, {f(ds['lesion_dice_D']['hi'])}]; median paired difference {f(ds['median_difference_common'])}."]
    if "unpaired_median_difference" in c:
        u = c["unpaired_median_difference"]; L.append(f"Unpaired difference of medians on each model's own set: {f(u['point'])} [{f(u['ci'][0])}, {f(u['ci'][1])}].")
    if "sensitivity_excluding_flagged" in c:
        s = c["sensitivity_excluding_flagged"]; L.append(f"Sensitivity excluding flagged edits: n {s['n']}, {f(s['point'])} [{f(s['ci'][0])}, {f(s['ci'][1])}] ({s['verdict']}).")
b = res["arm_B"]; L += ["", f"## Arm B (bright to dark): BRIGHT-50 {b['n_bright50']} lesions, D0 {b['n_d0']} (minimum {b['min_required']}): **no verdict**", ""]
for r in b["lesions"]:
    L.append(f"- {r['case']} rank {r['rank']}: BRIGHT-50 baseline coverage {r['bright50']['base_cov']:.2f}, delta_cov(edit) {f(r['bright50']['dcov_edit'])}, control {f(r['bright50']['dcov_ctrl'])}, D_B {f(r['bright50']['D'])}; D0 " + ("not an Arm B lesion (detected at baseline)" if r["d0"] is None else f"baseline {r['d0']['base_cov']:.2f}, delta_cov(edit) {f(r['d0']['dcov_edit'])}, D_B {f(r['d0']['D'])}"))
open(os.path.join(out, "m3_contrast.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
