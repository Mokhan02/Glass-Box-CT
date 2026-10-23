"""Side-by-side M1 table (trained activations vs raw-HU control vs random-init) per layer, from the analysis outputs. Usage: python scripts/w3s2_m1_report.py <analysis dir> [<analysis dir> ...]"""
import csv
import json
import sys

for d in sys.argv[1:]:
    rows = list(csv.DictReader(open(f"{d}/probe_table.csv"))); dec = json.load(open(f"{d}/decision.json")); head = json.load(open(f"{d}/head_check.json"))
    print(f"## {d}: n={dec['n_cases']} verdict {dec['primary']['verdict']}; median interior-FN margin {head['interior_FN_margin_median']:.2f} (IQR {head['interior_FN_margin_IQR'][0]:.2f} to {head['interior_FN_margin_IQR'][1]:.2f}); share within 1 unit below 0: {head['share_FN_within_1_unit_below_0']:.3f}")
    layers = []
    for r in rows:
        if r["layer"] not in layers:
            layers.append(r["layer"])
    for sub in ("pooled", "small tumours", "large tumours"):
        print(f"  [{sub}] probe A, interior-FN detection at 95% specificity (95% CI; patients; tokens)")
        for h in layers:
            def g(model, fs):
                m = [r for r in rows if r["model"] == model and r["layer"] == h and r["features"] == fs and r["probe"] == "A" and r["subset"] == sub]
                return f"{float(m[0]['interior_FN_rate']):.3f} [{float(m[0]['ci_low']):.3f}, {float(m[0]['ci_high']):.3f}]" if m and m[0]["interior_FN_rate"] not in ("", "nan") else "n/a"
            tr = [r for r in rows if r["model"] == "trained" and r["layer"] == h and r["features"] == "activations" and r["probe"] == "A" and r["subset"] == sub][0]
            print(f"    {h:12s} trained {g('trained', 'activations'):26s} raw-HU {g('trained', 'rawHU'):26s} random-init {g('random', 'activations'):26s} (n tokens {tr['interior_FN_tokens']}, patients {tr['n_patients_with_interior_FN']})")
    p = dec["primary"]; print(f"  decoder_F minus mid: {p['decoder_minus_layer'][0]:+.3f} [{p['decoder_minus_layer'][1]:+.3f}, {p['decoder_minus_layer'][2]:+.3f}]; late(a) {p['late_criterion_a']}, late(b) {p['late_criterion_b']}; {p['random_note']}")
