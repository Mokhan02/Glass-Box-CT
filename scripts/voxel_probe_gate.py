"""Gate for the readout re-fit (results/readout_refit/PREREG.md applies only if this passes). Mechanical; written before the clean-cohort and corrected-control runs finished.
(a) The governing verdict from the two clean cohorts (81 single-acquisition, 75 UID-route) is LOST LATE, or both say 'ENCODER REPRESENTS' and the median interior-FN logit of BOTH is <= -2.0.
(b) In BOTH clean cohorts, the trained mid-encoder interior-FN detection CI lower bound exceeds the CI upper bound of the corrected random-init control (same token groups, random activations).
Usage: python scripts/voxel_probe_gate.py results/voxel_probe"""
import csv
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/voxel_probe"
COH = {"single-acquisition (81)": ("sensitivity_single_acquisition",), "UID route (75)": ("sensitivity_uid_route",)}
short = lambda v: v.split(":")[0].strip()  # noqa: E731
lines = ["# Gate for the readout re-fit", "", "Rule fixed in `scripts/voxel_probe_gate.py` before the clean-cohort and corrected-control runs finished.", ""]
dec = {k: json.load(open(os.path.join(root, v[0], "decision.json"))) for k, v in COH.items()}
verd = {k: short(d["verdict"]) for k, d in dec.items()}; vs = list(verd.values())
governing = vs[0] if vs[0] == vs[1] else "INCONCLUSIVE (the clean cohorts disagree)"
lines += ["## (a) governing verdict from the clean cohorts", "", "| cohort | verdict | R_enc [95% CI] | R_hu [95% CI] | median interior-FN logit |", "|---|---|---|---|---|"]
for k, d in dec.items():
    lines.append(f"| {k} | {verd[k]} | {d['R_enc']:.3f} [{d['CI_enc'][0]:.3f}, {d['CI_enc'][1]:.3f}] | {d['R_hu']:.3f} [{d['CI_hu'][0]:.3f}, {d['CI_hu'][1]:.3f}] | {d['median_FN_logit']:.2f} |")
a_pass = governing.startswith("LOST LATE") or (all(v.startswith("ENCODER REPRESENTS") for v in vs) and all(d["median_FN_logit"] <= -2.0 for d in dec.values()))
lines += ["", f"Governing verdict: **{governing}** -> condition (a): **{'PASS' if a_pass else 'FAIL'}**", "", "## (b) trained mid-encoder vs corrected random-init control (probe A, interior-FN detection)", "",
          "| cohort | trained mid [95% CI] | random-init mid [95% CI] | trained CI lower > random CI upper? |", "|---|---|---|---|"]
b_ok = []
for k, v in COH.items():
    p = os.path.join(root, "random_control_corrected", v[0], "probe_table.csv")
    if not os.path.exists(p):
        lines.append(f"| {k} | corrected control missing | | FAIL |"); b_ok.append(False); continue
    R = list(csv.DictReader(open(p))); g = lambda m: next(r for r in R if r["model"] == m and r["layer"] == "mid" and r["features"] == "activations" and r["probe"] == "A" and r["subset"] == "pooled")  # noqa: E731
    t, r = g("trained"), g("random"); tl, ru = float(t["ci_low"]), float(r["ci_high"]) if r["ci_high"] not in ("", "nan") else float("nan")
    ok = tl > ru; b_ok.append(ok)
    lines.append(f"| {k} | {float(t['interior_FN_rate']):.3f} [{tl:.3f}, {float(t['ci_high']):.3f}] | {float(r['interior_FN_rate']):.3f} [{float(r['ci_low']):.3f}, {ru:.3f}] | {'yes' if ok else 'NO'} |")
b_pass = all(b_ok)
lines += ["", f"Condition (b): **{'PASS' if b_pass else 'FAIL'}** (must hold in both clean cohorts)", "", f"## GATE: **{'PASSED' if a_pass and b_pass else 'FAILED'}**" + ("" if a_pass and b_pass else "  -> skip the readout re-fit and record this in results/SUMMARY.md")]
open(os.path.join(root, "GATE.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
