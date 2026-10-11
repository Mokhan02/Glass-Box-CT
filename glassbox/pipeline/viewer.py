"""Browser-friendly copies of a scan: CT and masks resampled to a coarser isotropic grid (default 1.5 mm, the model's own working resolution), so a sample loads in seconds.
Measurements are made on the original grid; only the viewer files change. Stored voxel coordinates (lesions, L3) are remapped through world coordinates so click-to-locate still lands."""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def _remap(vox, old_affine, new_affine) -> list[int]:
    world = old_affine @ np.array([*vox, 1.0]); return [int(round(v)) for v in (np.linalg.inv(new_affine) @ world)[:3]]


def make_viewer_files(ct_path: str, scan_dir: str, mm: float = 1.5) -> dict:
    ct = nib.load(ct_path); old = ct.affine; out = {}
    new_ct = resample_to_output(ct, voxel_sizes=[mm] * 3, order=1, cval=-1024); data = np.clip(np.rint(np.asarray(new_ct.dataobj)), -1024, 3071).astype(np.int16)
    nib.save(nib.Nifti1Image(data, new_ct.affine), os.path.join(scan_dir, "ct.nii.gz")); new = new_ct.affine
    for name in ("organs", "tumour"):
        m = nib.load(os.path.join(scan_dir, f"{name}.nii.gz")); r = resample_to_output(m, voxel_sizes=[mm] * 3, order=0)
        nib.save(nib.Nifti1Image(np.asarray(r.dataobj).astype(np.uint8), r.affine), os.path.join(scan_dir, f"{name}.nii.gz")); assert np.allclose(r.affine, new), "viewer grids differ"
    les = os.path.join(scan_dir, "lesions.json"); L = json.load(open(les))
    for l in L["lesions"]:
        l["voxel_in_file"] = _remap(l["voxel_in_file"], old, new)
    L["viewer_grid_mm"] = mm; json.dump(L, open(les, "w"), indent=1)
    bcp = os.path.join(scan_dir, "body_composition.json"); B = json.load(open(bcp))
    if B.get("available"):
        B["l3_voxel_in_file"] = _remap(B["l3_voxel_in_file"], old, new); B["viewer_grid_mm"] = mm; json.dump(B, open(bcp, "w"), indent=1)
    return {"viewer_shape": list(data.shape), "viewer_mb": round(sum(os.path.getsize(os.path.join(scan_dir, f)) for f in ("ct.nii.gz", "organs.nii.gz", "tumour.nii.gz")) / 1e6, 1)}
