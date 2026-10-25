"""W5 sanity gate (PREREG): Task03 fold-0 validation liver Dice >= 0.93 and tumour Dice >= 0.597 (model D's 0.647 minus 0.05), from nnU-Net's own validation/summary.json ("mean" of labels 1 and 2).
Usage: python scripts/w5_gate.py <validation/summary.json> <out json>. Exit 0 on PASS, 1 on FAIL."""
import json
import sys

LIVER_MIN, TUMOUR_MIN = 0.93, 0.597
s = json.load(open(sys.argv[1])); liver, tumour = float(s["mean"]["1"]["Dice"]), float(s["mean"]["2"]["Dice"])
res = {"summary": sys.argv[1], "liver_dice": liver, "tumour_dice": tumour, "liver_min": LIVER_MIN, "tumour_min": TUMOUR_MIN, "liver_ok": liver >= LIVER_MIN, "tumour_ok": tumour >= TUMOUR_MIN}
res["pass"] = res["liver_ok"] and res["tumour_ok"]; json.dump(res, open(sys.argv[2], "w"), indent=1)
print(("PASS" if res["pass"] else "FAIL"), f"liver {liver:.4f} (>= {LIVER_MIN}) tumour {tumour:.4f} (>= {TUMOUR_MIN})")
sys.exit(0 if res["pass"] else 1)
