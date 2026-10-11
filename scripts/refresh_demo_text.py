"""Idempotent text-field migration for already-built result JSON (no inference, no measurement changes): applies the current wording of
the standing limitation and the contrast-basis fields from glassbox/pipeline/scan.py to each <demo_dir>/<scan>/lesions.json.
Usage: python scripts/refresh_demo_text.py [data/demo]"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")
from glassbox.agent.tools import qty  # noqa: E402
from glassbox.pipeline.scan import LIMITATION  # noqa: E402
from glassbox.pipeline.select import CARD  # noqa: E402
from glassbox.pipeline.viewer import align_to_viewer, make_outline_layers  # noqa: E402

for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "*", "lesions.json"))):
    d = json.load(open(f)); d["standing_limitation"] = LIMITATION
    t = [l for l in d["lesions"] if l["recist"] == "target candidate"]
    if "sum_of_target_diameters" not in d:  # older results only; the pipeline now writes it from the unrounded diameters
        d["sum_of_target_diameters"] = qty(round(sum(l["longest_axial_diameter"]["value"] for l in t), 1), "mm"); d["n_targets"] = len(t)
    if "tumour_liver_contrast" in d:
        d["contrast_basis"] = "mean HU of the predicted tumour minus mean HU of liver tissue outside a margin around it (research measurement)"; d["liver_reference_margin"] = qty(5, "mm")
    json.dump(d, open(f, "w"), indent=1); print("refreshed", f)

idx = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "index.json")  # landing-page card fields (chip + one-liner) from the selection slot
if os.path.exists(idx):
    d = json.load(open(idx))
    for sc in d["scans"]:
        sc["chip"], sc["oneliner"] = CARD[sc["selection"]]
    json.dump(d, open(idx, "w"), indent=1); print("refreshed", idx)

for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "demo_*"))):
    print("aligned to viewer grid" if align_to_viewer(f) else "already aligned", f)

for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "demo_*", "*.json"))):  # earlier versions stored the grid size as a bare float
    d = json.load(open(f))
    if isinstance(d, dict) and isinstance(d.get("viewer_grid_mm"), (int, float)):
        d["viewer_grid_mm"] = qty(d["viewer_grid_mm"], "mm"); json.dump(d, open(f, "w"), indent=1); print("unit added to viewer_grid_mm in", f)

for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "demo_*"))):
    print("outline layers", os.path.basename(f), make_outline_layers(f))
import re  # reference contrast / coverage for the scan page's one-line explanation, parsed from the blurb the pick stage wrote from the same numbers
idx = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "data/demo", "index.json")
d = json.load(open(idx))
for sc in d["scans"]:
    m = re.search(r"found (\d+)% of the largest tumour.*?tumour minus local liver ([+-]?\d+) HU", sc["blurb"])
    if m:
        sc["reference_coverage_pct"], sc["reference_contrast_hu"] = int(m.group(1)), int(m.group(2))
    sc["reference_default"] = sc["selection"] in ("bright_missed_flagged", "shift_matters")
json.dump(d, open(idx, "w"), indent=1); print("index fields added")
