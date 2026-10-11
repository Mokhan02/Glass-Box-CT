# Experiment 1 (part a) audit: post-processing, class head, RECIST split, prior art

CPU-only; nothing re-run on the GPU; nothing in the pre-registered voxel-probe analysis was touched. Everything below is quoted from files
(paths relative to this repo's `.venv/lib/python3.11/site-packages/` for MONAI 1.6.1, to the downloaded bundle, or to the downloaded
`nvidia/NV-Segment-CT` Hugging Face files). Where I am not certain, it says so.

## 1. Bundle post-processing

### (a) Binarisation threshold
There is **no sigmoid-then-0.5 step and no tunable, per-class threshold** anywhere in the shipped path. The network output is pre-sigmoid logits
(`monai/apps/vista3d/inferer.py:70`: "stitched_output: [1, B, H, W, D]. The value is before sigmoid."), and the post-transform acts on logits:

```
monai/apps/vista3d/transforms.py:155    pred[pred < 0] = 0.0
monai/apps/vista3d/transforms.py:163    pred[pred > 0] = 1.0        # single-prompt branch (object_num == 1)
```
A positive logit is foreground, so the effective threshold is logit > 0, i.e. sigmoid > 0.5, identical for every class. The model-zoo `configs/inference.json`
(lines 141-149) runs `monai.apps.vista3d.transforms.VistaPostTransformd` then `Invertd` with `"nearest_interp": true` (line 149) on the **label map**; the HF pipeline does the same
(`nvidia/NV-Segment-CT vista3d_pipeline.py:159` and `:164`). The VISTA3D paper's silence on the threshold is consistent with this: it is implicit in the code.
Difference from our own pipeline: we resample **sigmoid probabilities** with bilinear interpolation to the original grid and then apply p > 0.5 (`glassbox/seg/infer.py:45,51`),
whereas the shipped path thresholds first and resamples the label map with nearest neighbour. This can shift boundaries by a fraction of a voxel; it has not been quantified.

### (b) How overlapping positives are resolved
When several class prompts are run together, the label map is an **argmax over class logits, with no priority order**:
```
monai/apps/vista3d/transforms.py:157    if object_num > 1:
monai/apps/vista3d/transforms.py:159        is_bk = torch.all(pred <= 0, dim=0, keepdim=True)
monai/apps/vista3d/transforms.py:160        pred = pred.argmax(0).unsqueeze(0).float() + 1.0
monai/apps/vista3d/transforms.py:161        pred[is_bk] = 0.0
```
(negatives were clipped to 0 at line 155, so the class with the largest positive logit wins; a voxel where every prompted class has logit <= 0 is background). Liver does not
beat hepatic tumour by rule: it wins only where its logit is larger. Two further facts from the code:
- **"Segment everything" never outputs hepatic tumour.** `vista3d_pipeline.py:75-78`: `EVERYTHING_LABEL = list(set([i + 1 for i in range(132)]) - set([2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132]))`.
  Class 26 (hepatic tumour) must be requested explicitly. Classes 129 (kidney mass), 130 (liver tumour) and 131 cannot be prompted by label alone at all (`vista3d_pipeline.py:246-248`:
  `supported_list = list({i + 1 for i in range(132)} - {16, 18, 129, 130, 131})`), consistent with what I found earlier for C4KC-KiTS.
- **Our reported tumour masks involve no merge.** The mask is class 26's own logit or probability against the threshold: `scripts/save_pred_tumor.py:26-27` (prompt `[26]`, `probs[0] > 0.5`),
  `scripts/voxel_probe_extract.py:86` (`pred = logit > 0.0`), `scripts/eval_heldout.py:64-66` (each target thresholded on its own probability; multiple prompts for the *same* target are unioned, never argmaxed
  against another organ). **So the merge step cannot account for the under-segmentation we measured.** The shipped argmax rule could only remove voxels from such a mask.

**Measurement requested: fraction of missed GT-tumour voxels with tumour sigmoid above threshold that lost in the merge. NOT MEASURABLE from stored data.** What is stored for HCC predictions:
binary class-26 masks (103 cases), class-26 probability maps (55 of 103 at the time of writing; the voxel-probe extraction is still running), and sampled class-26 logits of interior GT-tumour voxels.
The liver (class 1) and all other class logits/probabilities were computed in memory and not kept. I did not re-run inference, as instructed. Given the previous paragraph the measured fraction would be 0 for our masks by construction;
what remains unknown is how many voxels the *shipped two-class argmax pipeline* would additionally lose, which needs the liver logit at those voxels.

### (c) Does the class-embedding MLP `M` have a bias term?
Yes inside `M`, but **the logit itself has no additive bias.**
```
monai/networks/nets/vista3d.py:604-609   self.mlp = nn.Sequential(nn.Linear(feature_size, feature_size), nn.InstanceNorm1d(1), nn.GELU(), nn.Linear(feature_size, feature_size))
monai/networks/nets/vista3d.py:610       self.class_embeddings = nn.Embedding(n_classes, feature_size)
monai/networks/nets/vista3d.py:634-639   src = self.image_post_mapping(src) ... class_embedding = self.mlp(class_embedding) ... masks_embedding = class_embedding.squeeze() @ src.view(b, c, h * w * d)
```
Checkpoint (`vista3d_pretrained_model/model.pt`) tensors: `class_head.mlp.0.weight (48,48)`, `class_head.mlp.0.bias (48,)`, `class_head.mlp.3.weight (48,48)`, `class_head.mlp.3.bias (48,)`,
`class_head.class_embeddings.weight (512,48)`, and four `class_head.image_post_mapping.*.conv*.weight (48,48,3,3,3)`. The per-class readout vector w_i = M(E_c[i]) is a fixed 48-vector (biases of `M` are folded into it),
and the logit at voxel v is exactly `w_i . F_v` with **no scalar offset**. Consequences for the planned experiments:
- A "bias-only re-fit" needs a **new** scalar parameter; it is mathematically a uniform threshold shift.
- **Feature width is 48, not 256.** Every class-head tensor in this checkpoint is 48-wide (and `decoder_F` measured (1, 48, 128, 128, 128)). The strategy document's "256-dimensional" statements do not apply to `vista3d132`; a re-fit readout vector is 48 numbers.
- The dotted feature F_v is the output of **two residual conv blocks with instance normalisation** (`vista3d.py:611-630`, `norm_name="instance"`), so F_v, and hence the logit, depends on statistics of the whole 128^3 window. That is a candidate mechanism for window-dependent logits that has not been tested.
- Exact decomposition of a logit over the 48 channels of F is available directly (`logit = sum_c w_c F_c`), no SAE needed. Our decoder_F SAE was trained on 12 mm **average-pooled tokens** (pool x8), not on per-voxel F, so it cannot decompose voxel logits as built.

## 2. RECIST results, split three ways
Source: `cache/pulled/recist_lesions.csv` (115 target lesions >= 10 mm, 103 cases, max 2 per case), written by `scripts/recist_shift.py`; computed by `scripts/exp1_recist_split.py`.
Tables: `recist_split_lesions.csv`, `recist_split_cases.csv` in this folder. CIs are case-level bootstrap (1000 resamples).

**How missed lesions were handled before (so the earlier numbers can be read correctly).**
- First run: "detected" = any overlap with a predicted component, so a single overlapping voxel counted; a lesion with no overlap was a miss.
- Visual check (`results` figure `recist_check.png`) showed a lesion counted as detected with a 0 mm prediction. The rerun uses **detected = predicted voxels cover >= 10% of the lesion's volume**;
  sub-threshold overlap is a miss.
- In both runs a missed lesion contributes **predicted diameter 0 mm** to the case's sum of diameters (relative error -100% for that lesion); the "median error (mm)" column used **detected lesions only**,
  while the "median relative error" column in the earlier size table counted misses as -100%. Those two columns therefore had different denominators; this split separates them.
- The predicted diameter of a detected lesion is the largest overlapping predicted component, so a fragmented prediction is measured by its biggest piece (biases the error negative).
- Ground-truth diameters come from the semi-automatic HCC-TACE-Seg masks, not from radiologist RECIST measurements.

**Lesion level (detected = >= 10% of volume covered):**

| GT diameter | n | detection rate [95% CI] | any overlap | detected only: median error (mm) [CI], n | detected only: median relative | combined median relative (miss = -100%) | share of mm shortfall from misses |
|---|---|---|---|---|---|---|---|
| 10-30 mm | 8 | 0.25 [0.00, 0.67] | 0.38 | -6.2 [-9.0, -3.3], n=2 | -0.29 | -1.00 | 91% |
| 30-60 mm | 43 | 0.60 [0.45, 0.75] | 0.72 | -9.4 [-13.0, -4.4], n=26 | -0.22 | -0.40 | 75% |
| 60-100 mm | 30 | 0.93 [0.83, 1.00] | 1.00 | -9.9 [-13.4, -5.9], n=28 | -0.12 | -0.15 | 35% |
| >= 100 mm | 34 | 0.88 [0.77, 0.97] | 0.97 | -21.2 [-33.6, -10.5], n=30 | -0.16 | -0.18 | 47% |
| all | 115 | 0.75 [0.67, 0.83] | 0.84 | -11.3 [-13.8, -8.7], n=86 | -0.17 | -0.25 | 57% |

**Case level (sum of target diameters, RECIST's quantity). A = misses counted as 0 mm (as in earlier reports). B = detected targets only.**

| cases | n | A: median relative error | A: >= 20% / >= 30% off | B: n, median relative error | B: >= 20% / >= 30% off | share of mm shortfall from misses |
|---|---|---|---|---|---|---|
| all | 103 | -0.24 | 0.52 / 0.44 | 79, -0.15 | 0.38 / 0.25 | 57% |
| GT sum 22-55 mm | 35 | -0.45 | 0.71 / 0.63 | 21, -0.21 | 0.52 / 0.38 | 73% |
| GT sum 55-101 mm | 35 | -0.20 | 0.49 / 0.43 | 29, -0.17 | 0.38 / 0.31 | 49% |
| GT sum 101-278 mm | 35 | -0.16 | 0.40 / 0.29 | 31, -0.13 | 0.32 / 0.16 | 51% |

**Reading it.** The error is one-sided in both parts, but the size dependence lives mostly in **detection**: small/mid lesions (30-60 mm) are missed 40% of the time, while
the bias among *detected* lesions is a moderate -12% to -22% relative with no clear trend across the larger bins. A constant relative bias would largely cancel in a percentage change between
timepoints; the detection component would not. The 10-30 mm bin has only 8 lesions (2 detected) and supports no conclusion. Not tested: whether any of this changes response categories (needs paired timepoints or a simulation).
Earlier wording to avoid: "30-60 mm lesions are under-measured by a median of 40%" mixes misses (as -100%) with measurement error; among detected 30-60 mm lesions the median is -22%.

## 3. Prior-art check: arXiv 2607.20993 (HTML version read)
The PDF has no extractable text, so I stopped there at first; you then asked for the HTML version, which has readable text (arxiv.org/html/2607.20993v1). **What follows is the fetch tool's extraction of that page, not
a reading of the full paper; verify any number against the paper before quoting it.**
- *Sparse Concept Channels in Frozen 3D CT Vision Encoders*; Nooralahzadeh, Bogensperger, Bluethgen, Krauthammer; 23 July 2026.
- **Models:** Pillar-0 (3D chest CT vision-language model, D=1152, frozen) and Merlin (3D abdominal CT, D=512, frozen); CT-CLIP/CT-CHAT numbers are cited, not rerun; Qwen3-VL-8B as a report-verbalizer baseline.
- **Data:** CT-RATE (chest; 18 findings; 1,564-volume validation split as test), RadChest-CT (external; 3,630 volumes, 84 labels), Merlin abdominal CT (30 findings). Liver appears only as an anatomical category in a circuit-overlap figure.
- **Method:** per-finding channel ranking by per-channel AUROC minus 0.5 on a calibration set, top K=10 kept, a closed-form mean-difference probe (no trained weights), ablation by zeroing the channels. Steering is not mentioned.
  Controls: random, zero-shot prompts, full-dimension probe, supervised linear heads, off-target ablation.
- **Claims (as extracted):** CCP-10 mean AUROC 0.798 vs 0.793 for all 1,152 dimensions (zero-shot 0.682; supervised head 0.833); ablating a finding's channels drops its AUROC by 0.0060 on-target vs 0.00003 off-target (about 20x);
  Merlin CCP-10 AUROC 0.839; a training-free report generator beats CT-CHAT on micro-F1 (0.549 vs 0.184).
- **Not in the paper (per the extraction):** sparse autoencoders, segmentation networks, SegResNet, VISTA-3D, RECIST, lesion size, detection-failure analysis.
**Overlap with this project: low.** The shared idea is locating and ablating internal units of a frozen 3D CT encoder. It differs in model family (vision-language classification encoders vs a segmentation network), task (finding classification and report text vs lesion
measurement), method (channel probing and zeroing vs SAEs and a linear segmentation readout) and data. Still not claimed: that anything here is "first"; a broader literature search was not done.

## 4. Discrepancies with the strategy document (to fix before relying on it)
- Feature width: 48, not 256 (above).
- "Bias-only re-fit" is a new parameter equal to a threshold shift (above); there is no existing bias to re-fit in the logit.
- The merge-rule explanation is ruled out for our reported tumour masks by construction; it is still unmeasured for the shipped two-class pipeline.
- Single-window analysis must account for instance-norm window dependence (`vista3d.py:611-630`).
- Whether the 132-class checkpoint saw HCC-TACE-Seg or C4KC-KiTS is **still unverified**: the model card lists no training data, and `data_license.txt` (232 bytes) names only the Medical Segmentation Decathlon.

## 5. Data-quality findings that affect the HCC cohort (found while preparing the lesion comparison)
All from DICOM headers and files already on the instance; nothing was re-run on the GPU. Scripts: `scripts/hcc_phase_check.py`, `scripts/hcc_duplicate_inspect.py`, `scripts/hcc_acq_check.py`.
- **22 of the 103 labelled CT series contain more than one acquisition** (19 with two, 3 with three; distinct `AcquisitionNumber`). 13 have several copies at identical slice positions (none pixel-identical; mean difference 27-64 HU);
  9 more are offset (irregular slice spacing). **All 22 are in the geometry route; none of the 75 UID-route series are.** My converter sorted every file of a series into a single volume, so for these 22 cases the CT given to the model interleaves acquisitions.
  The "SEG labels every other slice" pattern I earlier read as sparse annotation is, for the duplicate-position cases, an artifact of that: HCC_001's SEG labels 37 slices and its series has 38 unique positions.
- **This does not explain the label-route gap.** Tumour Dice: all 103 = 0.521; single-acquisition only (81) = 0.551; 22 mixed-acquisition geometry cases = 0.410 vs 6 single-acquisition geometry cases = 0.117 (n=6, very wide interval); UID route 0.585 (n=75).
- **Geometry-route labels have unverified alignment.** Their SEGs reference source instances that are absent from the collection (the "SOP instances ... not included in the Referenced Series Sequence" messages), so I matched them to a CT by physical position only.
  A small structure like a tumour can then be misaligned while the liver still overlaps well (liver Dice 0.91-0.94 in both routes). This is a hypothesis consistent with the data, not a finding.
- **The headline RECIST bias holds on the UID route alone, but the pooled sum-of-diameters number is route-sensitive:**

| subset | detection | detected-lesion error | case sum error, median [95% CI] | >= 20% off | overestimates >= 20% |
|---|---|---|---|---|---|
| all 103 | 0.75 | -11.3 mm (-17%) | -24% [-31%, -17%] | 52% | 0 |
| UID route (75) | 0.81 | -11.4 mm (-17%) | -19% [-28%, -13%] | 48% | 0 |
| geometry route (28) | 0.55 | -10.4 mm (-17%) | -50% | 64% | 0 |

  Quote the UID-route figures as the primary estimate; the one-sided bias among detected lesions (about -11 mm, -17%) is identical across subsets.
- **The pre-registered voxel-probe experiment includes these cases** (it reports uid and geometry separately, as pre-registered). It was not changed. A sensitivity analysis restricted to the 81 single-acquisition cases is an addition for you to decide; I have not edited `results/voxel_probe/`.


## 6. Corrected conversion of the geometry-route cases (`scripts/convert_heldout_clean.py`; per-case table `hcc_clean_conversion.csv`)
For each of the 28 geometry cases: find the CT series holding the SOP instances the SEG references (anywhere in the study), then keep an acquisition only if >= 95% of the SEG frame positions coincide (< 0.5 mm) with its instances.
Written to a new directory; the original conversion (read by the pre-registered voxel-probe extraction) was not modified.
- **14 RESOLVED** (one acquisition matches), **11 AMBIGUOUS** (two or three acquisitions at identical positions; one volume written per candidate), **3 UNRESOLVED and excluded**: HCC_015 and HCC_016 (referenced series found, but no acquisition's positions match the SEG frames) and HCC_048 (no referenced instance in the study).
- The first conversion used the wrong series for 5 cases (HCC_006, 015, 016, 104, 105).
- **Resolved cases (n = 14), old vs corrected conversion:** tumour Dice 0.327 -> 0.394 (+0.067 [-0.030, +0.198]); liver Dice 0.898 -> 0.917 (+0.019 [+0.007, +0.033]). Individual cases move both ways. **The corrected geometry cases (0.394) remain well below the UID route (0.585): the route gap is not an artifact of the interleaving.**
- **RECIST on the clean cohort (75 UID-route + 14 resolved = 89 cases, 101 lesions):** missed (< 10% covered) 23%; detected lesions -11.3 mm (-17%); sum-of-diameters error **-21% [-30%, -15%]**; 51% of cases off by >= 20%, 40% by >= 30%; **0 overestimates**.
  The 14 resolved cases alone: sum-of-diameters error -36% (it was -75% under the old conversion), 47% of lesions missed (53% before).
- **Ambiguous cases (11): the choice of acquisition changes tumour Dice a lot** (spread > 0.2 in 4 of 11, up to 0.75); see `results/exp8_lesions/SUMMARY.md` section 4.
- The evaluation of the clean volumes initially ran against the wrong root path (my error: `eval_heldout.py` takes the parent directory) and produced nothing; it was rerun correctly. Saved masks and RECIST used full paths and were unaffected.
