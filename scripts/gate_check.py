import sys, time, json, csv
import numpy as np, torch
sys.path.insert(0, ".")  # run from the repo root
from monai.transforms import LoadImage
from glassbox.seg import classes as C
from glassbox.seg.infer import load_network, predict_probs, dice

dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")
scan = sys.argv[1]
root = f"cache/ts/{scan}"
meta = json.load(open("cache/bundle_metadata.json"))
names = meta["network_data_format"]["outputs"]["pred"]["channel_def"]
inv = {v: int(k) for k, v in names.items() if k != "0"}
want = {"spleen": ["spleen"], "liver": ["liver"], "aorta": ["aorta"],
        "iliopsoas_left": ["left iliopsoas"], "iliopsoas_right": ["right iliopsoas"],
        "autochthon_left": ["left autochthon"], "autochthon_right": ["right autochthon"], "vertebrae_L3": ["vertebrae L3"]}
ids = {k: inv[v[0]] for k, v in want.items()}
print({k: i for k, i in ids.items()})
net = load_network("cache/weights/model.pt", dev)
t = time.time()
probs, m = predict_probs(f"{root}/ct.nii.gz", list(ids.values()), net, dev)
print("infer s:", round(time.time() - t), "probs", tuple(probs.shape))
loader = LoadImage(image_only=True)
res = {}
for j, (k, i) in enumerate(ids.items()):
    gt = loader(f"{root}/segmentations/{k}.nii.gz").numpy() > 0.5
    pr = probs[j].numpy() > 0.5
    if pr.shape != gt.shape: print("SHAPE MISMATCH", k, pr.shape, gt.shape); continue
    res[k] = round(dice(pr, gt), 4) if gt.sum() else None
print(scan, res)
json.dump(res, open(f"cache/dice_{scan}.json", "w"))
