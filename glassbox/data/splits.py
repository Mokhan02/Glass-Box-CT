"""Patient-level train/val/test split. Never split by slice or by scan within a patient."""
from __future__ import annotations

import random
from collections.abc import Iterable

SEED = 0


def split_patients(
    patient_ids: Iterable[str],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SEED,
) -> dict[str, str]:
    """Map each unique patient id to 'train', 'val' or 'test'. Deterministic for a given seed."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    ids = sorted(set(patient_ids))
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * fractions[0])
    n_val = round(len(ids) * fractions[1])
    out: dict[str, str] = {}
    for i, pid in enumerate(ids):
        out[pid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out
