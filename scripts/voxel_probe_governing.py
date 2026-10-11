"""Apply the disagreement rule from results/voxel_probe/DEVIATIONS.md #2, mechanically.
Reads decision.json from the primary run and the two clean-cohort runs; writes results/voxel_probe/GOVERNING.md."""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/voxel_probe"
runs = {"primary (all 103)": os.path.join(root, "decision.json"), "single-acquisition cohort": os.path.join(root, "sensitivity_single_acquisition", "decision.json"), "UID-route cohort": os.path.join(root, "sensitivity_uid_route", "decision.json")}
dec = {k: json.load(open(p)) for k, p in runs.items() if os.path.exists(p)}
short = lambda v: v.split(":")[0].strip()  # noqa: E731
verd = {k: short(d["verdict"]) for k, d in dec.items()}
clean = [verd.get("single-acquisition cohort"), verd.get("UID-route cohort")]
gov = clean[0] if clean[0] is not None and clean[0] == clean[1] else "INCONCLUSIVE (the two clean cohorts disagree or one is missing)"
prim = verd.get("primary (all 103)")
lines = ["# Governing verdict (rule fixed in DEVIATIONS.md #2)", "", "| run | verdict | R_enc [95% CI] | R_hu [95% CI] | decoder_F minus mid [95% CI] | median FN logit |", "|---|---|---|---|---|---|"]
for k, d in dec.items():
    lines.append(f"| {k} | {short(d['verdict'])} | {d['R_enc']:.3f} [{d['CI_enc'][0]:.3f}, {d['CI_enc'][1]:.3f}] | {d['R_hu']:.3f} [{d['CI_hu'][0]:.3f}, {d['CI_hu'][1]:.3f}] | {d['decoder_minus_mid'][0]:+.3f} [{d['decoder_minus_mid'][1]:+.3f}, {d['decoder_minus_mid'][2]:+.3f}] | {d['median_FN_logit']:.2f} |")
lines += ["", f"**Governing verdict: {gov}**", f"Primary verdict: {prim}." + ("" if prim == gov else " It DIFFERS from the governing verdict; per the pre-stated rule the clean cohorts govern interpretation.")]
open(os.path.join(root, "GOVERNING.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
