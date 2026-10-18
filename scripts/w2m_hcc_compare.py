"""Compare a re-run of the HCC probe analysis with a stored result set (probe_table.csv, decision.json): numeric differences, and whether any verdict or trained-vs-control comparison changes category.
Usage: python scripts/w2m_hcc_compare.py <repro_dir> <stored_dir> [label]   (prints a JSON summary)"""
import json
import os
import sys

import numpy as np
import pandas as pd

KEY = ["model", "layer", "features", "probe", "subset"]


def sep(t, layer, subset, ctrl_model, ctrl_features):
    """trained-vs-control separation on the primary quantity: trained activations CI lower > control CI upper, for probe A."""
    tr = t[(t.model == "trained") & (t.layer == layer) & (t.features == "activations") & (t.probe == "A") & (t.subset == subset)]
    ct = t[(t.model == ctrl_model) & (t.layer == layer) & (t.features == ctrl_features) & (t.probe == "A") & (t.subset == subset)]
    if tr.empty or ct.empty or ct.ci_high.isna().all():
        return None
    return bool(float(tr.ci_low.iloc[0]) > float(ct.ci_high.iloc[0]))


def compare(rd, sd, label=""):
    a, b = pd.read_csv(os.path.join(rd, "probe_table.csv")), pd.read_csv(os.path.join(sd, "probe_table.csv"))
    assert list(a.columns) == list(b.columns) and len(a) == len(b), "table shape differs"
    num = [c for c in a.columns if a[c].dtype != object]; D = (a[num].astype(float) - b[num].astype(float)).abs()
    out = {"label": label, "cells": int(D.notna().to_numpy().sum()), "cells_differing": int((D > 1e-9).to_numpy().sum()), "max_abs_diff": float(np.nanmax(D.to_numpy())), "max_abs_diff_mid": float(np.nanmax(D[(a.layer == "mid").to_numpy()].to_numpy()))}
    rate = ["interior_FN_rate", "ci_low", "ci_high", "TP_rate", "TP_ci_low", "TP_ci_high"]
    ra, rb = a[rate].astype(float).round(3), b[rate].astype(float).round(3); out["quoted_precision_cells_changed_3dp"] = int((~((ra == rb) | (ra.isna() & rb.isna()))).to_numpy().sum())      # NaN-vs-NaN is unchanged
    ch = [];
    for layer in ("mid", "bottleneck", "decoder_F"):
        for subset in sorted(a.subset.unique()):
            for cm, cf in (("trained", "rawHU"), ("random", "activations")):
                x, y = sep(a, layer, subset, cm, cf), sep(b, layer, subset, cm, cf)
                if x is not None or y is not None:
                    ch.append({"layer": layer, "subset": subset, "control": f"{cm}/{cf}", "stored": y, "repro": x, "changed": x != y})
    out["trained_vs_control_comparisons"] = len(ch); out["trained_vs_control_changed"] = [c for c in ch if c["changed"]]
    da, db = json.load(open(os.path.join(rd, "decision.json"))), json.load(open(os.path.join(sd, "decision.json")))
    out["verdict_stored"], out["verdict_repro"] = db["verdict"].split(":")[0], da["verdict"].split(":")[0]; out["verdict_same"] = out["verdict_stored"] == out["verdict_repro"]
    out["decision_booleans_same"] = all(da[k] == db[k] for k in ("represents", "never", "late_criterion_a", "late_criterion_b"))
    out["decision_numbers_3dp_same"] = {k: bool(np.round(np.array(da[k], dtype=float), 3).tolist() == np.round(np.array(db[k], dtype=float), 3).tolist()) for k in ("R_enc", "CI_enc", "R_hu", "CI_hu", "decoder_F_rate", "decoder_minus_mid", "median_FN_logit")}
    return out


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""), indent=1))
