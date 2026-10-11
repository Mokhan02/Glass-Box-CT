"""Dataset registry. `in_distribution` means the dataset was in VISTA-3D training
(paper, supplementary Table 1), so results on it must never be presented as held-out."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    in_distribution: bool
    note: str = ""


DATASETS: dict[str, DatasetInfo] = {
    d.name: d
    for d in [
        DatasetInfo("totalsegmentator", True, "labels partly model-derived"),
        DatasetInfo("msd_task03", True),
        DatasetInfo("msd_task06", True),
        DatasetInfo("msd_task07", True),
        DatasetInfo("msd_task08", True),
        DatasetInfo("msd_task09", True),
        DatasetInfo("msd_task10", True),
        DatasetInfo("abdomenct1k", True),
        DatasetInfo("amos22", True),
        DatasetInfo("verse", True),
        DatasetInfo("ct_org", True),
        DatasetInfo("tcia_pancreas_ct", True),
        # External-test sets in the paper; labels/license/access still to confirm.
        DatasetInfo("c4kc_kits", False, "CC BY 3.0; kidney and mass; kidney mass (129) needs point prompts"),
        DatasetInfo("adrenal_acc_ki67_seg", False, "dropped: only an adrenal-lesion label, which the model has no class for"),
        DatasetInfo("hcc_tace_seg", False, "CC BY 4.0; use the manual SEG only, not the AIMI AI segmentation"),
        DatasetInfo("word", False, "dropped: GPL-3.0, gated by email request, terms forbid building on it; not used"),
    ]
}


def is_in_distribution(dataset: str) -> bool:
    # Unknown datasets are treated as in-distribution: the safe default for claims.
    info = DATASETS.get(dataset)
    return True if info is None else info.in_distribution
