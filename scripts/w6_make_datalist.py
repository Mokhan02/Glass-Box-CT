"""W6: the MONAI-bundle data list of MSD Task03 for the fine-tune (PREREG W6 section 3). Fold assignment from nnU-Net's own fold-0 split (`splits_final.json` of Dataset003_Liver, as in W3 and W5): the nnU-Net fold-0 VALIDATION cases get "fold": 0 (the bundle validates on fold 0 and trains on every other fold),
all other labelled cases "fold": 1. Validation cases are ordered by a seeded shuffle (seed = first 8 hex digits of SHA-256 of `w6_val_order`), so that `n_val_samples=10` takes the same 10 cases in every arm. Image / label paths are relative to --dataset-dir (`imagesTr/<id>.nii.gz`, `labelsTr/<id>.nii.gz`).
Usage: python scripts/w6_make_datalist.py <dataset_dir with imagesTr/ labelsTr/> <splits_final.json> <out.json> [--fold 0]"""
import hashlib
import json
import os
import random
import sys


def build(dataset_dir, splits, fold=0):
    val = set(splits[fold]["val"]); train = set(splits[fold]["train"]); ids = sorted(train | val)
    for i in ids:
        for sub in ("imagesTr", "labelsTr"):
            assert os.path.exists(os.path.join(dataset_dir, sub, f"{i}.nii.gz")), (sub, i)
    order = sorted(val); random.Random(int(hashlib.sha256(b"w6_val_order").hexdigest()[:8], 16)).shuffle(order)
    rows = [{"image": f"imagesTr/{i}.nii.gz", "label": f"labelsTr/{i}.nii.gz", "fold": 0} for i in order] + [{"image": f"imagesTr/{i}.nii.gz", "label": f"labelsTr/{i}.nii.gz", "fold": 1} for i in sorted(train)]
    return {"training": rows, "n_validation": len(val), "n_train": len(train), "source": "nnU-Net splits_final.json fold %d" % fold}


if __name__ == "__main__":
    d = build(sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[sys.argv.index("--fold") + 1]) if "--fold" in sys.argv else 0)
    json.dump(d, open(sys.argv[3], "w"), indent=1); print(d["n_train"], "train,", d["n_validation"], "validation cases")
