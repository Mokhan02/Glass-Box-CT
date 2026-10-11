"""Run the full pipeline on the 89-case clean cohort, then pick the five demo scans by the stated rules (glassbox/pipeline/select.py).
Stages (instance, GPU):  --stage rebuild -> re-run the pipeline on the picked demo scans only (refreshes their result JSONs, keeps the viewer files)
                         --stage all   -> cache/demo_all/<case>/ result files for every clean-cohort case (no CT copied)
                         --stage pick  -> cache/demo/<demo_k>/ (viewer-sized CT + masks + results) and cache/demo/index.json
Reference-based facts (coverage, contrast) come from the counterfactual baseline table (cache/contrast_cf/baseline.csv); they are used only for choosing and describing demos."""
import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, ".")
from glassbox.pipeline.scan import RECAL_LOGIT, build_results  # noqa: E402
from glassbox.pipeline.select import CARD, select_demos  # noqa: E402
from glassbox.pipeline.viewer import make_outline_layers, make_viewer_files  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["all", "pick", "rebuild"]); ap.add_argument("--out", default="cache"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}; conv = {r["case"]: r["status"] for r in csv.DictReader(open("results/exp1_audit/hcc_clean_conversion.csv"))}
cohort = [(c, "uid") for c in sorted(acq) if acq[c]["route"] == "uid"] + [(c, "resolved") for c in sorted(conv) if conv[c] == "RESOLVED"]
if a.stage == "all":
    import torch

    from glassbox.pipeline.scan import infer_probs
    from glassbox.seg.infer import load_network
    dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev)
    for i, (case, src) in enumerate(cohort[: a.limit or None], 1):
        ct = os.path.join(ROOTS[src], case, "ct.nii.gz")
        build_results(ct, infer_probs(ct, net, dev), os.path.join(a.out, "demo_all"), case, {"source": "HCC-TACE-Seg (TCIA, CC BY 4.0)", "case": case, "in_distribution": False, "label_route": src, "model": "NV-Segment-CT vista3d132"})
        print(f"[{i}/{len(cohort)}] {case}", flush=True)
elif a.stage == "rebuild":
    import torch

    from glassbox.pipeline.scan import infer_probs
    from glassbox.seg.infer import load_network
    dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev)
    demo = os.path.join(a.out, "demo"); idx = json.load(open(os.path.join(demo, "index.json")))
    for it in idx["scans"]:
        case = it["source"]["case"]; src = dict(cohort)[case]; ct = os.path.join(ROOTS[src], case, "ct.nii.gz")
        build_results(ct, infer_probs(ct, net, dev), demo, it["id"], {"source": "HCC-TACE-Seg (TCIA, CC BY 4.0)", "case": case, "in_distribution": False, "label_route": src, "model": "NV-Segment-CT vista3d132"}, recal_logit=RECAL_LOGIT)
        make_viewer_files(ct, os.path.join(demo, it["id"])); make_outline_layers(os.path.join(demo, it["id"]))  # build_results wrote original-grid masks and voxel coordinates: put them back on the viewer grid
        print("rebuilt", it["id"], case, flush=True)
        if os.path.exists(os.path.join(demo, it["id"], "report.json")):
            os.remove(os.path.join(demo, it["id"], "report.json"))
else:
    shiftm = {}
    for r in csv.DictReader(open("results/readout_refit/bright_threshold_lesions.csv")):
        if r["case"] not in shiftm or float(r["diam_mm"]) > shiftm[r["case"]][0]:
            shiftm[r["case"]] = (float(r["diam_mm"]), float(r["cov_orig"]) < 0.10 and float(r["cov_shift"]) >= 0.5)
    base = {}
    for r in csv.DictReader(open("cache/contrast_cf/baseline.csv")):
        if r["rank"] == "1":
            base[r["case"]] = r
    rows = []
    for case, src in cohort:
        d = os.path.join(a.out, "demo_all", case)
        if case not in base or not os.path.exists(os.path.join(d, "quality_audit.json")):
            continue
        qa = json.load(open(os.path.join(d, "quality_audit.json")))["structures"][0]; les = json.load(open(os.path.join(d, "lesions.json"))); bc = json.load(open(os.path.join(d, "body_composition.json")))
        rows.append({"id": case, "src": src, "coverage": float(base[case]["coverage"]), "contrast": float(base[case]["contrast"]), "diam_mm": float(base[case]["diam_mm"]), "flag": qa["flag"], "assessed": bool(qa.get("assessed")),
                     "l3_available": bool(bc.get("available")), "n_lesions": les["n_lesions"], "p": qa.get("failure_probability"), "shift_matters": shiftm.get(case, (0, False))[1]})
    picks = select_demos(rows); index = []; demo = os.path.join(a.out, "demo"); shutil.rmtree(demo, ignore_errors=True); os.makedirs(demo)
    for k, p in enumerate(picks, 1):
        r = p["row"]
        if r is None:
            print(f"slot {p['slot']}: NO candidate satisfies the rule; left out (not substituted)"); continue
        sid = f"demo_{k}"; shutil.copytree(os.path.join(a.out, "demo_all", r["id"]), os.path.join(demo, sid))
        d = os.path.join(demo, sid)
        for f in ("report.json",):
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
        info = make_viewer_files(os.path.join(ROOTS[r["src"]], r["id"], "ct.nii.gz"), d)
        blurb = (f"{p['why']} Against the reference segmentation, the model found {100 * r['coverage']:.0f}% of the largest tumour (reference longest diameter {r['diam_mm']:.0f} mm; tumour minus local liver {r['contrast']:+.0f} HU). "
                 f"Source: HCC-TACE-Seg, case {r['id'][:7]} (The Cancer Imaging Archive, CC BY 4.0). Chosen by rule, not at random: see selection.")
        index.append({"id": sid, "title": f"{k}. {p['title']}", "blurb": blurb, "source": {"dataset": "HCC-TACE-Seg", "case": r["id"], "license": "CC BY 4.0"}, "selection": p["slot"], "chip": CARD[p["slot"]][0], "oneliner": CARD[p["slot"]][1], "reference_contrast_hu": int(round(r["contrast"])), "reference_coverage_pct": int(round(100 * r["coverage"])), "reference_default": p["slot"] in ("bright_missed_flagged", "shift_matters"), "viewer_mb": info["viewer_mb"]})
    json.dump({"scans": index, "selection_note": "Scans were chosen by fixed rules (glassbox/pipeline/select.py) to show typical successes, the main failure mode, and the flag's own false alarms. They are not a random sample."}, open(os.path.join(demo, "index.json"), "w"), indent=1)
    print(json.dumps([(i["id"], i["selection"], i["viewer_mb"]) for i in index]))
