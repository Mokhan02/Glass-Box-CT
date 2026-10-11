# Glass-Box CT

Interpreting NVIDIA's NV-Segment-CT (VISTA-3D) with probes and sparse autoencoders, wrapped in a Nemotron agent that measures body composition and lesions and flags when the segmentation is likely wrong.

**Research use only. Not for clinical decisions.** NV-Segment-CT weights carry a non-commercial license.

Status: scaffolding. Findings, setup steps and the "How NVIDIA models and Nebius are used" section will be filled in as the work lands.

Layout: `glassbox/` holds one subpackage per stage (`data`, `seg`, `extract`, `bodycomp`, `probes`, `sae`, `autointerp`, `agent`, `app`); `writeup/` holds figures and the post.

## Gotchas for anyone using NV-Segment-CT (`vista3d132`)

These come from reading the shipped code (quoted with file and line in `results/exp1_audit/AUDIT.md`), not from the paper.
- **"Segment everything" never outputs hepatic tumour.** Tumour and lesion classes (23 lung, 24 pancreatic, 26 hepatic, 27 colon, 128 bone lesion) are excluded from that mode, so an automatic run on a liver CT will show no tumour whether or not one exists. Prompt class 26 explicitly. This repository's tools enforce it (`glassbox/seg/prompts.py`).
- **Classes 129 (kidney mass) and 130 (liver tumour) cannot be prompted by label.** They need point prompts.
- **The foreground threshold is logit > 0 (probability 0.5) for every class**, applied before resampling the label map. There is no tunable threshold in the shipped pipeline.
- **With several classes prompted together, overlaps are resolved by argmax over the class logits**, with no priority order.
- **On held-out HCC CT the tumour class under-measures lesions, always in the same direction.** See `results/` and the write-up before using it for RECIST-style measurements.

