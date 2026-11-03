"""W4 (WAW-TACE) split of the patients over the two A10 instances (results/w4_waw/PREREG.md): a seeded random permutation (seed = first 8 hex digits of the SHA-256 of `w4_waw_instance_split`) of the patient ids found in
patient_qa.csv, cut so that the two halves hold 100 / 99 patients (every patient contributes two series, so the series counts are 200 / 198 at most). Writes patients_half0.txt and patients_half1.txt.
Usage: python scripts/w4_waw_split.py <patient_qa.csv> <out dir>"""
import csv
import hashlib
import os
import sys

import numpy as np


def split(ids):
    ids = sorted(set(ids), key=int); rng = np.random.default_rng(int(hashlib.sha256(b"w4_waw_instance_split").hexdigest()[:8], 16)); p = [ids[i] for i in rng.permutation(len(ids))]; h = (len(p) + 1) // 2
    return sorted(p[:h], key=int), sorted(p[h:], key=int)


if __name__ == "__main__":
    a, b = split([r["patient"] for r in csv.DictReader(open(sys.argv[1]))]); os.makedirs(sys.argv[2], exist_ok=True)
    for i, h in enumerate((a, b)):
        open(os.path.join(sys.argv[2], f"patients_half{i}.txt"), "w").write("\n".join(h) + "\n")
    print(len(a), len(b))
