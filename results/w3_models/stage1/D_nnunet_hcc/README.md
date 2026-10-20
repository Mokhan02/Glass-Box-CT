# Model D (nnU-Net, fold 0) on the HCC cohort: STOPPED by the Amendment 2 alignment check; no tumour analysis run

nnU-Net default inference (`nnUNetv2_predict`, `checkpoint_final.pth`): preprocessing as trained (CT clip and z-score, resampled to 1.0 x 0.77 x 0.77 mm), sliding window with step 0.5 and Gaussian weighting, test-time mirroring over all axis flips, probabilities averaged, argmax, resampled back to the input grid. **No post-processing was applied** (`nnUNetv2_find_best_configuration` was never run, so there is no `postprocessing.pkl`; nothing like model A's shipped connected-component filtering).

Alignment check (75 cases, the same code as model A; organ = nnU-Net label >= 1): **6 cases fail (limit 2)** -> the pre-registered rule is STOP and report. Per-case values: `w3_alignment_D_hcc.json`; diagnostic (no tumour metric): `gate_diag.json`, `scripts/w3_modelD_gate_diag.py`.

| case | scored organ Dice / centroid offset | predicted-organ components, share of volume outside the largest | largest component only: Dice / offset | reading |
|---|---|---|---|---|
| HCC_028_198668 | 0.815 / 45.8 mm | 2, 23% | 0.938 / 3.6 mm | stray non-liver blob; liver itself placed correctly |
| HCC_070_401358 | 0.905 / 23.0 mm | 2, 13% | 0.971 / 0.7 mm | stray blob |
| HCC_074_379020 | 0.855 / 22.1 mm | 3, 11% | 0.903 / 4.9 mm | stray blobs |
| HCC_090_076872 | 0.887 / 25.4 mm | 7, 14% | 0.962 / 1.3 mm | stray blobs |
| HCC_022_228695 | 0.692 / 29.2 mm | 4, 0.2% | 0.691 / 29.4 mm | genuine: liver largely missed (very large tumour; model A also under-segments this case, offset 13.4 mm) |
| HCC_077_996191 | 0.656 / 35.8 mm | 1, 0% | 0.656 / 35.8 mm | genuine: single component, liver largely mis-segmented |

Four of the six failures come from spurious predicted-organ blobs away from the liver (nnU-Net default has no connected-component step) and would pass on the largest component; two (HCC_022, HCC_077) are real liver failures. Nothing here changes the stop rule; the researcher decides. No tumour metric of model D has been computed or analysed (the lesion table `w3_lesions_D_nnunet_hcc.csv` was written by the scorer but not analysed).
