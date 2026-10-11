# Glass-Box CT

Interpreting NVIDIA's NV-Segment-CT (VISTA-3D) with probes and sparse autoencoders, wrapped in a Nemotron agent that measures body composition and lesions and flags when the segmentation is likely wrong.

**Research use only. Not for clinical decisions.** NV-Segment-CT weights carry a non-commercial license.

Status: scaffolding. Findings, setup steps and the "How NVIDIA models and Nebius are used" section will be filled in as the work lands.

Layout: `glassbox/` holds one subpackage per stage (`data`, `seg`, `extract`, `bodycomp`, `probes`, `sae`, `autointerp`, `agent`, `app`); `writeup/` holds figures and the post.
