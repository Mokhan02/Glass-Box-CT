# W4 (WAW-TACE) pre-specification of the contrast-sign and contrast-magnitude rules (scouting; fixed 2026-10-29 BEFORE any contrast is computed on WAW-TACE and before any model detection outcome exists on it)

Status when written: the WAW-TACE tumour masks and metadata have been inventoried (counts only); the CT archives are still downloading; no tumour contrast on any WAW-TACE phase, no registration and no model output exists. These rules are fixed now so that the counts below cannot be tuned to give a larger effective N. This is scouting, not a PREREG; the full PREREG (if GO) will repeat and extend it.

## Why
Masks were drawn on the most conspicuous phase, so the masked phase is a biased view of each tumour. The key paired test must separate the **sign** of the tumour-minus-liver contrast from its **magnitude**: a tumour that is bright in arterial and dark in portal venous with similar |contrast| in both phases lets the sign effect on detection be estimated with the magnitude held (nearly) fixed.

## Definitions (per tumour = one 26-connected lesion component of at least 10 mm longest axial diameter on the phase it was drawn on)
- c_A and c_P: tumour-minus-liver contrast (HU) on the arterial (phase 1) and portal-venous (phase 2) series, measured with the project's shared function (`glassbox.audit.contrast`, 5 to 20 mm liver shell, same fallback rule), on the drawn mask (its own phase) and on the mask propagated to the other phase by the registration that passes the QA threshold (to be proposed after the 5-patient check and fixed before the cohort run). Tumours whose registration fails QA are not counted.
- **Sign flip:** c_A and c_P have opposite signs AND both |c| >= delta.
- **Comparable |contrast|:** both |c| >= delta AND min(|c_A|, |c_P|) / max(|c_A|, |c_P|) >= rho.
- **Effective N for the sign-versus-magnitude test:** tumours with a sign flip AND comparable |contrast| (the tumours that vary sign at near-constant magnitude). Also reported: all tumours with a sign flip; all tumours with comparable |contrast| whatever the sign.
- **Primary setting: delta = 5 HU, rho = 0.5.** (delta keeps measurement noise near 0 HU from being counted as a flip; rho = 0.5 means the larger |contrast| is at most twice the smaller.)
- **Sensitivity grid, reported in full, nothing chosen from it:** delta in {0, 5, 10} HU x rho in {0.33, 0.5, 0.67}.

## Reporting rules fixed now
1. The counts (sign flips, comparable, both) are reported for the primary setting and for every cell of the grid, per registration-QA stratum (pass / fail) and overall, with the number of tumours lost to QA failure.
2. The primary setting and grid are not changed after the counts are seen. If the primary effective N is too small to be useful, that is a NO-GO or a redesign decision for the researcher, not a reason to retune rho or delta.
3. Planned detection model (for the PREREG, not run in scouting): detected (coverage >= 0.10) ~ sign + |contrast| + log(diameter) with patient clustering (GEE or cluster bootstrap); the sign coefficient is the quantity of interest, |contrast| and size are covariates. The contrast of tumours near zero is noisy; the delta dead zone is therefore part of the model's inclusion rule too (|c| < delta in either phase: reported as an exclusion count).
4. Liver reference for the contrast: the dataset's own TotalSegmentator liver mask of the same phase (label 5; TotalSegmentator is not one of the models under test, and VISTA-3D's training used its pseudo-labels for anatomy, which only affects the reference-liver definition and not the detection outcome); sensitivity: the liver mask eroded by 3 mm, and the median instead of the mean HU of the shell.
