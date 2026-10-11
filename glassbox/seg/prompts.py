"""Which class prompts a report must send to NV-Segment-CT (`vista3d132`).

GOTCHA: the model's "segment everything" mode silently omits tumour and lesion classes. In the Hugging Face pipeline
(nvidia/NV-Segment-CT, vista3d_pipeline.py lines 75-78):
    EVERYTHING_LABEL = list(set([i + 1 for i in range(132)]) - set([2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132]))
so hepatic tumour (26) is never produced unless it is prompted explicitly. Classes 16, 18, 129 (kidney mass), 130 (liver tumour) and 131
cannot be prompted by label alone at all (same file, lines 246-248); they need point prompts.
Anything that measures lesions must therefore prompt the tumour classes by index, and must not report "no tumour found" for a class it never asked about.
"""
from __future__ import annotations

from glassbox.seg import classes as C

EVERYTHING_EXCLUDED = (2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132)
NOT_PROMPTABLE_BY_LABEL = (16, 18, 129, 130, 131)
# Promptable tumour / lesion classes that "segment everything" leaves out (by name in the bundle metadata):
TUMOUR_PROMPTS = (C.LUNG_TUMOR, C.PANCREATIC_TUMOR, C.HEPATIC_TUMOR, 27, C.BONE_LESION)  # 27 = colon cancer primaries
NEEDS_POINT_PROMPTS = (C.KIDNEY_MASS, C.LIVER_TUMOR)  # 129, 130: no automatic output possible


def everything_prompts() -> list[int]:
    return [i for i in range(1, 133) if i not in EVERYTHING_EXCLUDED]


def report_prompts() -> list[int]:
    """Everything the automatic head can give by label, including the explicitly prompted tumour classes."""
    return everything_prompts() + list(TUMOUR_PROMPTS)


def require_tumour_prompts(prompted, needed=(C.HEPATIC_TUMOR,)) -> None:
    missing = [c for c in needed if c not in set(prompted or [])]
    if missing:
        raise ValueError(f"tumour classes {missing} were not prompted; 'segment everything' omits them, so an empty result would not mean 'no tumour'")
