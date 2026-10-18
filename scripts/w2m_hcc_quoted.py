"""Check the HCC voxel-probe numbers quoted in results/SUMMARY.md section 2 (and the README) against a re-run, at the quoted precision (3 decimals).
Quoted cells: probe A, pooled, interior-FN detection and its 95% CI for trained activations, random-init activations (corrected control) and trained raw-HU, at mid / bottleneck / decoder_F (all-103), and the mid-layer
trained and raw-HU rows for the single-acquisition and UID-route cohorts.
Usage: python scripts/w2m_hcc_quoted.py <repro runs dir> results/voxel_probe"""
import json
import sys

import pandas as pd

runs, V = sys.argv[1:3]
sets = {"all-103": (f"{runs}/primary_default_run2", V, f"{runs}/primary_corrected", f"{V}/random_control_corrected/primary"),
        "single-acquisition": (f"{runs}/single_acq", f"{V}/sensitivity_single_acquisition", f"{runs}/single_acq_corrected", f"{V}/random_control_corrected/sensitivity_single_acquisition"),
        "UID-route": (f"{runs}/uid", f"{V}/sensitivity_uid_route", f"{runs}/uid_corrected", f"{V}/random_control_corrected/sensitivity_uid_route")}


def cell(t, model, layer, feat):
    r = t[(t.model == model) & (t.layer == layer) & (t.features == feat) & (t.probe == "A") & (t.subset == "pooled")]
    return None if r.empty or pd.isna(r.interior_FN_rate.iloc[0]) else tuple(round(float(r[c].iloc[0]), 3) for c in ("interior_FN_rate", "ci_low", "ci_high"))


rows, changed = [], 0
for name, (rp, sp, rcp, scp) in sets.items():
    rt, st = pd.read_csv(f"{rp}/probe_table.csv"), pd.read_csv(f"{sp}/probe_table.csv"); rct, sct = pd.read_csv(f"{rcp}/probe_table.csv"), pd.read_csv(f"{scp}/probe_table.csv")
    for layer in ("mid", "bottleneck", "decoder_F"):
        for label, (a, b), model, feat in (("trained activations", (rt, st), "trained", "activations"), ("trained raw-HU", (rt, st), "trained", "rawHU"), ("random-init activations (corrected control)", (rct, sct), "random", "activations")):
            x, y = cell(a, model, layer, feat), cell(b, model, layer, feat)
            if x is None and y is None:
                continue
            same = x == y; changed += (not same); rows.append({"set": name, "layer": layer, "quantity": label, "stored": y, "rerun": x, "same_at_3dp": same})
out = {"quoted_cells_compared": len(rows), "quoted_cells_changed_at_3dp": changed, "cells": rows}
print(json.dumps({k: v for k, v in out.items() if k != "cells"}))
for r in rows:
    if not r["same_at_3dp"]:
        print("CHANGED", r)
json.dump(out, open(f"{runs}/quoted_check.json", "w"), indent=1)
