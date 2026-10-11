"""Browser-friendly copies of a scan: CT and masks resampled to a coarser isotropic grid (default 1.5 mm, the model's own working resolution), so a sample loads in seconds.
Measurements are made on the original grid; only the viewer files change. Stored voxel coordinates (lesions, L3) are remapped through world coordinates so click-to-locate still lands."""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output


def _remap(vox, old_affine, new_affine) -> list[int]:
    world = old_affine @ np.array([*vox, 1.0]); return [int(round(v)) for v in (np.linalg.inv(new_affine) @ world)[:3]]


def make_viewer_files(ct_path: str, scan_dir: str, mm: float = 1.5) -> dict:
    ct = nib.load(ct_path); old = ct.affine; out = {}
    new_ct = resample_to_output(ct, voxel_sizes=[mm] * 3, order=1, cval=-1024); data = np.clip(np.rint(np.asarray(new_ct.dataobj)), -1024, 3071).astype(np.int16)
    nib.save(nib.Nifti1Image(data, new_ct.affine), os.path.join(scan_dir, "ct.nii.gz")); new = new_ct.affine
    for name in ("organs", "tumour", "tumour_recal"):
        if not os.path.exists(os.path.join(scan_dir, f"{name}.nii.gz")):
            continue
        m = nib.load(os.path.join(scan_dir, f"{name}.nii.gz")); r = resample_to_output(m, voxel_sizes=[mm] * 3, order=0)
        nib.save(nib.Nifti1Image(np.asarray(r.dataobj).astype(np.uint8), r.affine), os.path.join(scan_dir, f"{name}.nii.gz")); assert np.allclose(r.affine, new), "viewer grids differ"
    les = os.path.join(scan_dir, "lesions.json"); L = json.load(open(les))
    for l in L["lesions"]:
        l["voxel_in_file"] = _remap(l["voxel_in_file"], old, new)
    L["viewer_grid_mm"] = {"value": mm, "unit": "mm"}; json.dump(L, open(les, "w"), indent=1)
    bcp = os.path.join(scan_dir, "body_composition.json"); B = json.load(open(bcp))
    if B.get("available"):
        B["l3_voxel_in_file"] = _remap(B["l3_voxel_in_file"], old, new); B["viewer_grid_mm"] = {"value": mm, "unit": "mm"}; json.dump(B, open(bcp, "w"), indent=1)
    return {"viewer_shape": list(data.shape), "viewer_mb": round(sum(os.path.getsize(os.path.join(scan_dir, f)) for f in ("ct.nii.gz", "organs.nii.gz", "tumour.nii.gz")) / 1e6, 1)}


def align_to_viewer(scan_dir: str) -> bool:
    """Put the masks on the viewer CT's grid and remap stored voxel coordinates to it (through world coordinates). Idempotent: does nothing once lesions.json
    carries `viewer_grid_mm`. Needs only the files in `scan_dir` (the viewer CT plus masks and result JSON that are still on the original grid), so it can repair a scan
    whose results were rebuilt after `make_viewer_files` ran. Returns True if anything changed."""
    les_p = os.path.join(scan_dir, "lesions.json"); L = json.load(open(les_p))
    if "viewer_grid_mm" in L:
        return False
    ct = nib.load(os.path.join(scan_dir, "ct.nii.gz")); old = nib.load(os.path.join(scan_dir, "tumour.nii.gz")).affine; mm = round(float(abs(ct.affine[0, 0])), 2)
    for name in ("organs", "tumour", "tumour_recal"):
        f = os.path.join(scan_dir, f"{name}.nii.gz")
        if not os.path.exists(f):
            continue
        m = nib.load(f)
        if m.shape == ct.shape and np.allclose(m.affine, ct.affine):
            continue
        r = resample_from_to(m, (ct.shape, ct.affine), order=0, cval=0); nib.save(nib.Nifti1Image(np.asarray(r.dataobj).astype(np.uint8), ct.affine), f)
    for l in L["lesions"]:
        l["voxel_in_file"] = _remap(l["voxel_in_file"], old, ct.affine)
    L["viewer_grid_mm"] = {"value": mm, "unit": "mm"}; json.dump(L, open(les_p, "w"), indent=1)
    bcp = os.path.join(scan_dir, "body_composition.json"); B = json.load(open(bcp))
    if B.get("available"):
        B["l3_voxel_in_file"] = _remap(B["l3_voxel_in_file"], old, ct.affine); B["viewer_grid_mm"] = {"value": mm, "unit": "mm"}; json.dump(B, open(bcp, "w"), indent=1)
    return True
