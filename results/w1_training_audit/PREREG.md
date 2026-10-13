# Pre-registration W1: training-data contrast audit

Status: APPROVED with edits (2026-10-12), to be committed on branch `research/roadmap` BEFORE any download or run. Deviations go to `results/w1_training_audit/DEVIATIONS.md` with reason and timing. Licence and size records made after this commit go to `results/w1_training_audit/DATASETS.md`, not into this file. All analyses run on CPU. Nothing on this branch changes the deployed behaviour of `main`.

## Question
VISTA-3D's liver-tumour class (26) misses tumours that are brighter than the surrounding liver (`results/SUMMARY.md` sections 3 and 6). Is that because bright liver tumours are rare among the liver-tumour labels the model was trained on?

W1 measures the contrast sign of public liver-tumour training labels. It does not show that the model learned anything from them (that is W5), and it cannot separate "training labels are rarely bright" from the two named confounders below (contrast phase, tumour type).

## Datasets (VISTA-3D paper, arXiv 2406.05285, Table 1)
VISTA-3D used 11,454 CT volumes, split 64/16/20% train/validation/test per source. Only sources with liver-tumour labels can enter this audit.

| Source (paper's name and row) | Liver-tumour labels | Labelled cases | Licence | Size | Role |
|---|---|---|---|---|---|
| MSD Task03 Liver ("MSD Task 3,6,7,8,9,10 [3]", 945 volumes) | yes: label 2 tumour, label 1 liver | 131 (training set; the 70 test cases are unlabelled) | CC-BY-SA 4.0 (AWS Open Data registry for MSD); confirm from `dataset.json` at download | recorded at download in DATASETS.md | **primary** |
| CRML-CT [52] ("Liver, colorectal liver metastases", 197 volumes) | yes | 197 | CC BY 4.0 (TCIA collection page) | 10.91 GB (TCIA page) | **primary** |
| MSD Task08 HepaticVessel (same MSD row) | tumour label only, **no liver label** | 303 (the 140 test cases are unlabelled) | as MSD | recorded at download | **secondary** (pseudo-liver, see below) |

Identification check (done before this approval): the paper's Table 1 row reads "CRML-CT [52] | Liver, colorectal liver metastases | 197", and reference [52] is Simpson et al., "Preoperative CT and survival data for patients undergoing resection of colorectal liver metastases", Scientific Data 11, 172 (2024). The TCIA Colorectal-Liver-Metastases collection has 197 subjects and the same title family and content. The paper never writes a DOI for [52], so the link to the TCIA collection rests on the matching count (197) and title; if the first download shows a different case count or content, stop, log it, and treat CRML-CT as unverified and secondary, leaving MSD Task03 as sole primary.

Other checks, logged if they fail:
- The MSD task sizes 131 + 63 + 281 + 303 + 41 + 126 = 945 match the paper's 945, which supports 131 and 303; confirm from `dataset.json`.
- The paper, as read, states no train/validation/test case lists. If the NV-Segment-CT bundle publishes them, the primary analysis uses training cases only and all labelled cases is a sensitivity analysis. If not (expected), all labelled cases are used and this is stated.
- Excluded: every other Table 1 source (no liver-tumour labels) and MSD tasks 6, 7, 9, 10.
- Not covered: VISTA-3D's final stage used MAISI synthetic tumours (count not given in the paper). Whether the MAISI generator's contrast can be sampled on CPU is checked separately; if not, it is a stated limit and nothing is claimed about the synthetic stage.

## Unit and lesion selection (identical to the HCC-TACE-Seg analysis)
Source of the definition: `scripts/exp8_lesion_covariates.py`, used by `results/contrast_cf/PREREG.md`.
- A lesion is a 3D connected component (26-connectivity) of the tumour label.
- Longest axial diameter = maximum Feret diameter over axial slices, using the in-plane spacing. Keep lesions >= 10 mm.
- **Primary unit: the 2 largest such lesions per case** (the HCC RECIST-target rule). Secondary: all lesions >= 10 mm.
- HCC used annotated slices only; the training sets are annotated through the whole volume, so the whole volume is used.

## Contrast definition (identical to HCC-TACE-Seg)
- T = mean HU of the lesion's voxels (raw HU, no windowing).
- L = mean HU of non-tumour liver voxels in the 5-20 mm shell around the lesion (3D Euclidean distance with true voxel spacing, from the lesion's own component).
- Fallback: if the shell has fewer than 200 voxels, L = mean HU of all non-tumour liver voxels.
- **Signed contrast c = T - L. Bright = c > 0 HU.**
- Liver mask: Task03 label 1 (tumour voxels are never in L). CRML-CT: the DICOM-SEG liver segment minus all tumour segments (checked on the first cases). Task08: pseudo-liver below.
- The 200-voxel fallback is 675 mm^3 at the HCC 1.5 mm grid but a different volume at other spacings. Sensitivity analyses: a 675 mm^3 volume threshold instead, and dropping all fallback lesions.
- The HCC extraction code is refactored into one function taking (CT in HU, tumour mask, liver mask, spacing) and used for the training sets. It is validated by (a) a synthetic-phantom unit test with known contrast, and (b) a reproduction test on HCC cases against `lesion_table.csv` `contrast_hu` (tolerance 0.1 HU), which needs the HCC volumes (not on this machine; on the Lambda instance). If (b) cannot be run before the primary result, say so in the summary; the HCC reference values come from the saved file in either case.

### Task08 pseudo-liver (secondary analysis, fixed now)
- Source: TotalSegmentator (Wasserthal et al.) `total` task, `liver` class, full resolution (not `--fast`), CPU, package version and weights hash recorded in DATASETS.md. This is the same family of tool that produced VISTA-3D's own pseudo-labels.
- Mask: keep the largest connected component of the predicted liver; remove all tumour-label voxels; use it as the liver mask in the contrast definition above.
- Lesions whose component lies entirely further than 10 mm from the pseudo-liver are excluded and counted. Lesions using the fallback are counted.
- If TotalSegmentator's licence terms or CPU run time make this impractical, Task08 is dropped and logged. It is never part of the primary result and every Task08 number is labelled "pseudo-liver".

## Comparison cohort
- **Primary HCC reference: the 75 UID-route cases** (86 lesions) from `results/exp8_lesions/lesion_table.csv` (`route == uid`), which exp8 saved under exactly this lesion and contrast definition (lesion-level shell contrast, 2 largest lesions >= 10 mm). Reused as is. All out of distribution. Context quoted with it: sum-of-diameters median bias **-19% [-28%, -13%]** (the UID route).
- **Sensitivity, later (logged as a planned follow-up, not part of the verdict):** the 89-case clean cohort (sum-of-diameters bias **-21% [-30%, -15%]**) once the 14 re-resolved cases are re-extracted from the corrected volumes on the Lambda instance.
- The existing 36 bright / 65 dark lesion split in `results/SUMMARY.md` section 5 uses a different, case-level contrast; it is not used here. Recalibration numbers, where mentioned, are the nested ones (`results/SUMMARY.md` section 7).

## Outputs
1. Per dataset and pooled: cases, lesions, c quantiles (5/25/50/75/95), share bright (c > 0) with a patient-level bootstrap 95% CI (1000 resamples of cases).
2. The same for the HCC reference.
3. Bright share by size stratum: <30, 30-60, >60 mm, in each cohort.
4. Figure: c distributions, training sets vs HCC.
5. Sensitivity: bright as c > +10 HU; fallback lesions dropped; the 675 mm^3 fallback; training split only (if lists exist); all lesions >= 10 mm; Task08 pseudo-liver added.

## Decision rule (fixed)
p_train = share of lesions with c > 0 in the **pooled** primary training audit (Task03 + CRML-CT, 2 largest per case, case-level bootstrap). p_HCC = the same on the primary HCC reference. **d = p_HCC - p_train**, with a 95% CI from independent case-level bootstraps of the two shares. **Margin = 0.10**, set by the roughly 86-lesion HCC sample (a share near 0.3 on about 86 lesions has a CI half-width of roughly 0.10, so a smaller margin could not be resolved). It is fixed here and not tuned.

The training side needs at least 200 lesions; otherwise the verdict is INCONCLUSIVE (not enough data). Otherwise apply, in this order, stopping at the first that holds:
1. **BRIGHT RARE IN TRAINING** (data-level cause supported): CI lower bound > 0 and d >= 0.10.
2. **REVERSED:** CI upper bound < 0.
3. **NO GAP:** the whole CI lies within [-0.10, +0.10].
4. **INCONCLUSIVE:** anything else.

The verdict is for the pooled primary analysis. Per-dataset results (Task03 alone, CRML-CT alone) are secondary, reported with the same rule, and if they point in opposite directions that is stated beside the verdict. Size strata and every sensitivity analysis are reported even when they disagree; they do not change the verdict.

## Named confounders and limits, stated in advance
- **Contrast phase.** CRML-CT is stated portal-venous; Task03 (from LiTS) and Task08 are mostly portal-venous but not guaranteed. HCC-TACE-Seg mixes acquisitions, and phase cannot be labelled there (the acquisition-time and contrast fields are stripped; the aorta/portal heuristic agreed with reference contrast no better than chance, kappa 0.04). A bright-share gap therefore cannot be split into labels versus phase mix. W4 and W5 address that.
- **Tumour type (metastases vs HCC).** CRML-CT is colorectal metastases, usually hypodense in portal-venous phase; Task03 is a mix of primary and secondary tumours; HCC-TACE-Seg is HCC, which enhances in arterial phase and washes out later. A low bright share in the training data is partly biology, not label selection. Per-dataset results show how much each source contributes.
- The audit shows what the labelled data look like, not what the model learned or how it weights them; VISTA-3D's exact training split and the MAISI synthetic stage may not be reproducible.
- Mean HU of a lesion includes necrosis and calcification and is a coarse summary of enhancement; it is the same measure as the HCC analysis, which is the point.
- Scanners, kernels and slice thicknesses differ; HU is compared raw.

## Planning note for W2 (not part of W1)
C4KC-KiTS is on the Lambda persistent filesystem at `/lambda/nfs/CTGate/`, not on this machine. Nothing is fetched now.

## Procedure
1. Commit this file. 2. Record licence, version and file sizes in `DATASETS.md`; stop on any mismatch listed above. 3. Download Task03 and CRML-CT (Task08 only after the primary analysis, and only if TotalSegmentator is usable). 4. Implement the shared function and its tests. 5. Run, then write `results/w1_training_audit/SUMMARY.md`.
