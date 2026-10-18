# W2-M M1 write-up: does the HCC tumour concept transfer to kidney tumour tissue?

Numbers: [m1_results.md](m1_results.md) / [m1_results.json](m1_results.json), produced by `scripts/w2m_m1_analyze.py` (commit e190275, unchanged). Rule and confounds: [../PREREG.md](../PREREG.md). Deviations attached: [../DEVIATIONS.md](../DEVIATIONS.md) #1, #1a and #3. C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K).

## Verdict (pre-registered three-tier rule): ORGAN-SPECIFIC
The HCC-trained tumour probe, applied unchanged to kidney mid-encoder tokens, has AUROC 0.455 [0.398, 0.517] for interior tumour tokens vs kidney-parenchyma tokens: at chance, not significantly below it. Both transfer controls score higher (HCC-trained random-init 0.692 [0.651, 0.724], raw-HU 0.652 [0.614, 0.688]), so the trained transfer beats neither, and the fraction of the kidney-native above-chance signal is -0.10 (SHARED needs 0.70). A kidney-native probe separates the same tokens well (0.964 [0.955, 0.971]; its random-init and raw-HU controls 0.823 and 0.909), so kidney tumour tissue is linearly separable at the mid-encoder; the HCC-derived direction just does not find it.

## Limits on what a failed transfer can show
- **The HCC source probe was only moderately sensitive on the hard tokens in its own domain.** In-domain on HCC the same probe (trained, mid, probe A, pooled) detects 0.587 [0.551, 0.656] of the interior tumour tokens the model MISSED at 95% specificity (`results/voxel_probe/probe_table.csv`); it detects 0.902 [0.877, 0.928] of the tumour tokens the model found, and its AUROC for missed tumour tokens vs liver is 0.893 [0.876, 0.914]. So the source probe is strong at telling tumour from liver overall and moderate on missed tissue; the figure of about 0.59 is that detection rate, not an AUROC. A single linear direction with moderate in-domain sensitivity on the hard subset is a modest bar for transfer, which limits what the failure to transfer can show.
- A fixed linear probe can fail across organs for reasons other than the absence of a shared concept: different feature statistics per organ and acquisition (kidney scans are arterial/corticomedullary with bright cortex), and a different tumour-versus-parenchyma contrast. M1 does not separate these explanations.
- The kidney-native result is largely carried by local HU (raw-HU native 0.909 vs trained native 0.964), so "separable at the mid-encoder" does not by itself mean the encoder holds a tumour concept beyond intensity.
- The "detection at the unchanged HCC-derived threshold" row (trained 0.785) is reported without the kidney-parenchyma false-positive rate at that threshold and must not be read as discrimination (DEVIATIONS #3).
- Tokens are 12 mm cells: 6 of 209 main-set lesions (1 bright, 5 dark) have no mid tumour token.
- The HCC reproduction in this environment was near-exact, not exact (DEVIATIONS #1 and #1a).

## Named confounds
Possible in-distribution images (AbdomenCT-1K, tumour labelling there unknown); arterial/corticomedullary phase (bright cortex inside the parenchyma reference); point prompts give location (the encoder tokens are computed without the point).
