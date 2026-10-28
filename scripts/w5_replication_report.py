"""W5 Amendment 2 replication report. Usage: python scripts/w5_replication_report.py <w5_results.json of BRIGHT-50> <w5_results.json of BRIGHT-50-s2 (its BRIGHT arm is labelled B50 by w5_analyze.py)> <outdir>
Replicates = the unchanged W5 verdict rule on BRIGHT-50-s2 (against D0 and SHAM-50) gives CONFIRMED (detection and Dice) again. Descriptive, not part of the verdict: |gap(B50) - gap(B50-s2)| for detection and Dice, the only seed-variance estimate available for the augmented arm (no D0-seed2 was run, so there is none for the baseline).
D0 and SHAM-50 must be identical in the two inputs (same cohort and data); otherwise the report refuses."""
import json
import os
import sys

a, b, outdir = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]; os.makedirs(outdir, exist_ok=True)
for arm in ("D0", "S50"):
    if a["arms"][arm] != b["arms"][arm]:
        raise SystemExit(f"arm {arm} differs between the two analyses; refusing")
ba, bb = a["arms"]["B50"], b["arms"]["B50"]
res = {"label": "SINGLE-SEED PER ARM (two BRIGHT seeds, one run each; no D0-seed2 was run, so no baseline variance estimate)", "replicates": b.get("prediction") == "CONFIRMED (detection and Dice)", "s2_prediction": b.get("prediction"), "s2_verdicts": b.get("verdicts"),
       "s2_contrasts": b.get("contrasts"), "b50_prediction": a.get("prediction"),
       "gaps": {"B50_seed_20261020": {"detection_gap": ba["detection_gap"], "dice_gap": ba["dice_gap"], "detection_bright": ba["detection_bright"], "mean_dice_all": ba["mean_dice_all"], "median_signed_sld_error": ba["median_signed_sld_error"]},
                "B50_s2_seed_20261028": {"detection_gap": bb["detection_gap"], "dice_gap": bb["dice_gap"], "detection_bright": bb["detection_bright"], "mean_dice_all": bb["mean_dice_all"], "median_signed_sld_error": bb["median_signed_sld_error"]}},
       "seed_variance_descriptive": {"abs_detection_gap_difference": abs(ba["detection_gap"] - bb["detection_gap"]), "abs_dice_gap_difference": abs(ba["dice_gap"] - bb["dice_gap"])}}
json.dump(res, open(os.path.join(outdir, "w5_replication.json"), "w"), indent=1)
lines = [f"# W5 Amendment 2: BRIGHT-50 replication ({res['label']})", "", f"- BRIGHT-50 (seed 20261020): {a.get('prediction')}; detection gap {ba['detection_gap']:+.3f}, Dice gap {ba['dice_gap']:+.3f}",
         f"- BRIGHT-50-s2 (seed 20261028): {b.get('prediction')}; detection gap {bb['detection_gap']:+.3f}, Dice gap {bb['dice_gap']:+.3f}", f"- **Replicates (CONFIRMED (detection and Dice) again): {res['replicates']}**",
         f"- Seed variance, descriptive: |detection gap difference| = {res['seed_variance_descriptive']['abs_detection_gap_difference']:.3f}, |Dice gap difference| = {res['seed_variance_descriptive']['abs_dice_gap_difference']:.3f}"]
open(os.path.join(outdir, "w5_replication.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
