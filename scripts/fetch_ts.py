"""Fetch selected TotalSegmentator v2.0.1 scans from the Zenodo zip by HTTP range requests (no 23 GB download).
Usage: python scripts/fetch_ts.py --dest cache/ts --ids s0235 s0291   |   --split train --n 400 [--seed 0]
Always writes meta.csv. Labels fetched: only the classes in LABELS (muscle, L3, big organs)."""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import time
import random

from remotezip import RemoteZip

URL = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
LABELS = ["spleen", "liver", "aorta", "iliopsoas_left", "iliopsoas_right",
          "autochthon_left", "autochthon_right", "vertebrae_L3"]


def retry(fn, what: str, tries: int = 10, base: float = 15.0):
    """Zenodo answers 429 when hit too fast. Back off (15 s, 30 s, ... capped at 5 min) instead of dying."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # RemoteIOError, connection resets, timeouts
            if i == tries - 1:
                raise
            wait = min(base * 2**i, 300)
            print(f"retry {i + 1}/{tries - 1} for {what} in {wait:.0f}s: {str(e)[:90]}", flush=True)
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="cache/ts")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="k/n: handle every n-th scan starting at k (run n processes in parallel)")
    ap.add_argument("--no-labels", action="store_true", help="CT only (for extraction runs)")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    z = retry(lambda: RemoteZip(URL), "zip index")
    with z:
        meta = retry(lambda: z.read("meta.csv"), "meta.csv").decode("utf-8-sig")
        open(os.path.join(a.dest, "meta.csv"), "w").write(meta)
        rows = list(csv.DictReader(io.StringIO(meta), delimiter=";"))
        ids = list(a.ids)
        if a.split:
            pool = sorted(r["image_id"] for r in rows if r["split"] == a.split)
            random.Random(a.seed).shuffle(pool)
            ids += pool[: a.n]
        k, n = map(int, a.shard.split("/"))
        ids = ids[k::n]
        for i, sid in enumerate(ids, 1):
            done = os.path.join(a.dest, sid, ".done")
            if os.path.exists(done):
                continue  # a scan counts as fetched only once every file landed (partial files are redone)
            files = ["ct.nii.gz"] + ([] if a.no_labels else [f"segmentations/{c}.nii.gz" for c in LABELS])
            for f in files:
                out = os.path.join(a.dest, sid, f)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                def get(f=f, out=out):
                    with z.open(f"{sid}/{f}") as src, open(out + ".part", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(out + ".part", out)

                retry(get, f"{sid}/{f}")
            open(done, "w").close()
            print(f"[shard {k}/{n}] {i}/{len(ids)} {sid}", flush=True)

if __name__ == "__main__":
    main()
