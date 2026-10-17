"""W2 v2 pilot gate (PREREG_v2.md). Reads w2v2_pilot_kidney.csv and w2v2_pilot_hcc.csv; applies the lesion rule (predicted/lesion volume < 3x and inside share > 0.3) and the arm rule
(>= 80% of pilot lesions pass and every size stratum present >= 60%). Exit code 0 only if all three PRIMARY arms pass; the secondary arm never blocks.
Usage: python scripts/w2v2_gate.py <dir with the pilot CSVs> <outdir>"""
import csv
import json
import sys

sys.path.insert(0, ".")
from glassbox.seg.w2_arms import ARMS, arm_passes, lesion_passes  # noqa: E402

d, out = sys.argv[1:3]
rows = [dict(r) for f in ("w2v2_pilot_kidney.csv", "w2v2_pilot_hcc.csv") for r in csv.DictReader(open(f"{d}/{f}"))]
for r in rows:
    r["pred_over_lesion_volume"] = float(r["pred_over_lesion_volume"]); r["pred_frac_in_lesion"] = float(r["pred_frac_in_lesion"])
res = {}
for arm, cfg in ARMS.items():
    rs = [r for r in rows if r["arm"] == arm]; res[arm] = {**arm_passes(rs), "role": cfg["role"], "median_volume_ratio": sorted(r["pred_over_lesion_volume"] for r in rs)[len(rs) // 2],
                                                         "median_inside_share": sorted(r["pred_frac_in_lesion"] for r in rs)[len(rs) // 2],
                                                         "lesions": [{"case": r["case"], "rank": int(r["rank"]), "group": r["group"], "stratum": r["stratum"], "diam_mm": float(r["diam_mm"]), "volume_ratio": r["pred_over_lesion_volume"],
                                                                      "inside_share": r["pred_frac_in_lesion"], "passes": lesion_passes(r["pred_over_lesion_volume"], r["pred_frac_in_lesion"])} for r in rs]}
primary_ok = all(res[a]["passes"] for a, c in ARMS.items() if c["role"] == "primary"); res["all_primary_arms_pass"] = primary_ok
json.dump(res, open(f"{out}/w2v2_gate.json", "w"), indent=1)
for arm in ARMS:
    r = res[arm]; print(f"{arm} ({r['role']}): {'PASS' if r['passes'] else 'FAIL'}  lesions passing {100 * r['pass_fraction']:.0f}% of {r['n']}; strata " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in r["strata"].items()) +
                         f"; median volume ratio {r['median_volume_ratio']:.2f}x, median inside share {r['median_inside_share']:.3f}")
print("ALL THREE PRIMARY ARMS PASS" if primary_ok else "AT LEAST ONE PRIMARY ARM FAILED: stop after the pilot and report with overlays")
sys.exit(0 if primary_ok else 2)
