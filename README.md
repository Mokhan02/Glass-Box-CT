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


## Data attribution (HCC-TACE-Seg, CC BY 4.0)
The sample scans in the demo are from the HCC-TACE-Seg collection on The Cancer Imaging Archive (TCIA), licensed CC BY 4.0. They are public, de-identified research CT, shown here only as research examples.
- Dataset: Moawad, A. W., Fuentes, D., Morshid, A., Khalaf, A. M., Elmohr, M. M., Abusaif, A., Hazle, J. D., Kaseb, A. O., Hassan, M., Mahvash, A., Szklaruk, J., Qayyom, A., Elsayes, K. (2021). *Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg)* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.5FNA-0924
- Data paper: Moawad, A. W. et al. (2023). Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation. *Scientific Data* 10, 33. https://doi.org/10.1038/s41597-023-01928-3
- TCIA: Clark, K. et al. (2013). The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository. *Journal of Digital Imaging* 26(6), 1045-1057. https://doi.org/10.1007/s10278-013-9622-7
The scans were resampled for viewing; the segmentations shown are this model's output, not the dataset's annotations.
