"""Turn class-probability maps into the result files the agent and the viewer use. GPU inference is separate (`infer_probs`); everything else runs on CPU and is unit-tested.
Writes, per scan, under <out>/<scan_id>/: segmentation.json, body_composition.json, lesions.json, quality_audit.json, organs.nii.gz, tumour.nii.gz, labels.json.
NOTE the model's 'segment everything' mode omits tumour classes, so class 26 is always prompted explicitly (glassbox/seg/prompts.py)."""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np

from glassbox.agent.tools import qty
from glassbox.bodycomp.lesions import measure_lesions
from glassbox.bodycomp.muscle import l3_muscle
from glassbox.quality.flag import assess, tumour_features
from glassbox.seg import classes as C
from glassbox.seg.prompts import require_tumour_prompts

ORGANS = {"liver": C.LIVER, "spleen": 3, "aorta": 6, "right kidney": C.RIGHT_KIDNEY, "left kidney": C.LEFT_KIDNEY, "vertebra L3": C.VERTEBRA_L3, "left iliopsoas": C.LEFT_ILIOPSOAS,
          "right iliopsoas": C.RIGHT_ILIOPSOAS, "left autochthon": C.LEFT_AUTOCHTHON, "right autochthon": C.RIGHT_AUTOCHTHON}
MUSCLE_KEYS = {"vertebrae_L3": C.VERTEBRA_L3, "iliopsoas_left": C.LEFT_ILIOPSOAS, "iliopsoas_right": C.RIGHT_ILIOPSOAS, "autochthon_left": C.LEFT_AUTOCHTHON, "autochthon_right": C.RIGHT_AUTOCHTHON}
PROMPTED = sorted(ORGANS.values()) + [C.HEPATIC_TUMOR]
MIN_ORGAN_CM3 = 1.0
# Held-out validation of the hepatic-tumour class, quoted in every lesion result so a report cannot omit the known bias (results/SUMMARY.md; UID route, 75 cases).
VALIDATION = {"cohort": "HCC-TACE-Seg held-out CT, 75 cases with verifiably linked labels (research data, not clinical)", "detected_lesion_median_error": qty(-11.4, "mm"),
              "detected_lesion_median_relative_error": qty(-17, "%"), "case_sum_of_diameters_median_error": qty(-19, "%"), "overestimates_of_20_percent_or_more": 0,
              "note": "On held-out data the hepatic-tumour class measured lesions smaller than the reference in every case where error was large; about one lesion in five was largely missed."}


def infer_probs(ct_path: str, net, device) -> dict[int, np.ndarray]:
    """GPU step: sigmoid probabilities for every prompted class in the CT's own geometry (hepatic tumour included, prompted explicitly)."""
    from glassbox.seg.infer import predict_probs
    probs, _ = predict_probs(ct_path, PROMPTED, net, device)
    return {c: probs[i].numpy() for i, c in enumerate(PROMPTED)}


def canonical(arr: np.ndarray, affine: np.ndarray):
    img = nib.as_closest_canonical(nib.Nifti1Image(np.ascontiguousarray(arr), affine)); return np.asarray(img.dataobj), tuple(float(z) for z in img.header.get_zooms()[:3])


def organ_label_map(probs: dict[int, np.ndarray]) -> np.ndarray:
    """Shipped rule: a class is foreground where its logit > 0 (p > 0.5); overlaps go to the class with the larger logit (monai VistaPostTransformd)."""
    names = list(ORGANS.values()); stack = np.stack([probs[c] for c in names]); lab = stack.argmax(0) + 1; lab[stack.max(0) <= 0.5] = 0; return lab.astype(np.uint8)


def build_results(ct_path: str, probs: dict[int, np.ndarray], out_dir: str, scan_id: str, provenance: dict) -> dict:
    img = nib.load(ct_path); affine, ct = img.affine, np.asarray(img.dataobj).astype(np.float32); zooms = tuple(float(z) for z in img.header.get_zooms()[:3]); vox = float(np.prod(zooms))
    d = os.path.join(out_dir, scan_id); os.makedirs(d, exist_ok=True)
    organs = organ_label_map(probs); tumour = probs[C.HEPATIC_TUMOR] > 0.5
    nib.save(nib.Nifti1Image(organs, affine), os.path.join(d, "organs.nii.gz")); nib.save(nib.Nifti1Image(tumour.astype(np.uint8), affine), os.path.join(d, "tumour.nii.gz"))
    json.dump({"organs": {str(i + 1): n for i, n in enumerate(ORGANS)}, "tumour": {"1": "hepatic tumour (class 26)"}}, open(os.path.join(d, "labels.json"), "w"))
    prov = {"provenance": provenance, "research_use_only": True}
    structures = []
    for i, (name, cls) in enumerate(ORGANS.items(), 1):
        v = float((organs == i).sum() * vox / 1000.0)
        if v >= MIN_ORGAN_CM3:
            structures.append({"name": name, "volume": qty(round(v, 1), "cm3")})
    seg = {**prov, "structures_found": [s["name"] for s in structures], "structures": structures, "prompted_classes": PROMPTED, "mask_files": ["organs.nii.gz", "tumour.nii.gz"],
           "note": "Tumour classes are prompted explicitly; the model's segment-everything mode omits them."}
    # body composition at L3 (canonical orientation: axis 2 axial)
    masks = {k: canonical((organs == list(ORGANS.values()).index(c) + 1).astype(np.uint8), affine)[0] for k, c in MUSCLE_KEYS.items()}
    cc, zc = canonical(ct, affine); m = l3_muscle(cc, masks, zc) if masks["vertebrae_L3"].any() else None
    if m is None:
        bc = {**prov, "available": False, "reason": "the L3 vertebra or enough muscle at L3 was not found in this scan's field of view"}
    else:
        bc = {**prov, "available": True, "l3_slice_index": int(m["l3_slice"]), "psoas_paraspinal_area": qty(round(m["muscle_area_cm2"], 1), "cm2"), "psoas_area": qty(round(m["psoas_area_cm2"], 1), "cm2"),
              "mean_muscle_density": qty(round(m["muscle_mean_hu"], 1), "HU"), "low_attenuation_fraction": round(m["lama_fraction"], 3),
              "label": "research measurement; psoas and paraspinal muscle only, not total skeletal muscle area, so not comparable with published sarcopenia thresholds"}
    # lesions
    tc, _ = canonical(tumour.astype(np.uint8), affine); pc, _ = canonical(probs[C.HEPATIC_TUMOR].astype(np.float32), affine)
    les = measure_lesions(tc.astype(bool), pc, zc); require_tumour_prompts(PROMPTED)
    lesions = {**prov, "prompted_classes": PROMPTED, "organ": "liver (hepatic tumour class 26)", "validation_context": VALIDATION,
               "lesions": [{"id": l["id"], "organ": "liver", "longest_axial_diameter": qty(round(l["diameter_mm"], 1), "mm"), "volume": qty(round(l["volume_cm3"], 1), "cm3"), "mean_probability": round(l["mean_probability"], 3),
                            "recist": l["recist"], "slice_index": l["best_slice"], "centroid_voxel": [int(round(c)) for c in l["centroid_voxel"]]} for l in les], "n_lesions": len(les)}
    if not les:
        lesions["note"] = "No hepatic tumour predicted. That is not the same as no tumour: on held-out data about one lesion in five was largely missed."
    # quality audit (tumour class only; organs are not assessed)
    a = assess(tumour_features(probs[C.HEPATIC_TUMOR], vox))
    qa = {**prov, "structures": [{"name": "hepatic tumour", **{k: (v if not isinstance(v, float) else round(v, 3)) for k, v in a.items() if k != "drivers"},
                                  **({"drivers": [{**dd, "weight_fraction": round(dd["weight_fraction"], 2)} for dd in a["drivers"]]} if a.get("assessed") else {})}],
          "organs_not_assessed": "Liver, kidney and other organ failures were too rare in the held-out data to fit or validate a flag, so no flag is given for them."}
    for name, obj in (("segmentation", seg), ("body_composition", bc), ("lesions", lesions), ("quality_audit", qa)):
        json.dump(obj, open(os.path.join(d, f"{name}.json"), "w"), indent=1)
    return {"segmentation": seg, "body_composition": bc, "lesions": lesions, "quality_audit": qa}
