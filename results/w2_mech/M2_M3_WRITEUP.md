# W2-M M2 and M3 write-up: erasure and brightness counterfactual on the kidney arm

Numbers: [m2/m2_results.md](m2/m2_results.md), [m3/m3_results.md](m3/m3_results.md) (committed scripts, unchanged). Rule and confounds: [PREREG.md](PREREG.md); deviations/notes: [DEVIATIONS.md](DEVIATIONS.md) #4 (and #1, #1a, #3 for the HCC reproduction and M1). Kidney arm = label 133 + point label 1 at the lesion's interior point, main-set lesions. C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K). The M2 branch (kidney-native eraser) and the M3 projection direction (kidney-native probe) follow from M1 being ORGAN-SPECIFIC.

## M2 verdict (pre-registered rule): NOT NECESSARY
Kidney-native rank-1 LEACE at the mid-encoder (fitted on training folds, applied to held-out patients; 209 lesions, 201 cases):
- (1) mean lesion Dice change -0.015 [-0.028, -0.002]. The CI excludes zero, so the effect is detectable, but it is far from the -0.05 margin: not met.
- (2) beyond the controls: LEACE minus random -0.017 [-0.032, -0.004], LEACE minus HU -0.020 [-0.036, -0.005]; the CI uppers are above -0.02: not met. The random-direction control changes Dice by +0.003 [+0.001, +0.004] and the HU-concept control by +0.005 [-0.007, +0.017].
- (3) the kidney organ Dice holds: +0.001 [-0.002, +0.005] on the seeded 60-case subset (unhooked mean Dice 0.932).
- Secondary failure modes: under kidney-native erasure LEAK rises 21% -> 27% (+6.2 pts [+2.9, +9.8]) and UNDER falls 33% -> 28% (-4.8 pts [-8.7, -1.4]); the random and HU controls move LEAK by +0.5 and -0.5 pts. Lesion coverage rises +0.022 [+0.012, +0.031].
- By brightness (descriptive): the Dice drop is in dark lesions (-0.018 [-0.034, -0.001]); bright lesions -0.005 [-0.019, +0.010].
- HCC-direction erasure (secondary, no controls): -0.016 [-0.025, -0.008] Dice, coverage -0.014; LEAK +2.4 pts [0.0, +4.8].
Reading: erasing the kidney-native tumour-vs-parenchyma direction at the mid-encoder has a small but real effect on the kidney arm (slightly less precise, more leakage), but nothing like the HCC result (lesion Dice -0.33), and no more than about 2 Dice points beyond a random or HU direction. By this rule the direction is not shown to be necessary; the organ itself is untouched.

## M3 verdict (pre-registered rule): PARTIAL
Primary: change in LEAK rate, edit minus sign-keeping control, paired per lesion (LEAK = predicted/lesion volume >= 3x or inside share <= 0.3):
- **Arm A (dark -> bright, 136 lesions, 132 cases): +24.3 pts [+17.8, +31.9]: meets** (CI lower > 0, point >= +10). LEAK goes from 0% (baseline, by construction) and 1% (control) to 26% after brightening; UNDER 42% -> 53% (control) -> 38% (edit). Dice edit minus control -0.101 [-0.145, -0.058]; coverage +0.102 [+0.076, +0.128] (over-segmentation).
- **Arm B (bright -> dark, 19 lesions): 0.0 pts [0.0, 0.0]: does not meet.** All 19 bright lesions were LEAK at baseline and stay LEAK under the edit and under the control (median predicted/lesion volume 7.0x baseline, 6.2x edit, 6.2x control; Dice +0.003 [-0.036, +0.047]).
- So brightening a dark kidney lesion makes the model leak beyond it far more than the matched sign-keeping control; darkening a bright leaking lesion does not undo the leak at these edit sizes.
- **EXPLORATORY projection** (kidney-native tumour direction, mid-encoder, lesion tokens, edited minus unedited): Arm A -1.05 [-1.23, -0.87] (brightening lowers the tumour score, 133 lesions with tokens); Arm B +0.24 [+0.09, +0.42] (18 lesions). Spearman of the projection change with the Dice change is about zero (A -0.08 [-0.24, +0.08], B -0.13 [-0.60, +0.42]).

## What M2 and M3 together do and do not show
- Brightening causes leakage (M3 Arm A), and it lowers the kidney-native tumour score (exploratory), but erasing that direction produces only a small change (M2), and the size of the projection change does not track the size of the Dice change (exploratory). So there is no support here for the brightness effect running mainly through this linear mid-encoder direction; this is a statement about this direction and this layer, not about the encoder.
- The kidney failure is mostly LEAK while the HCC failure was missed tumour; M1 (organ-specific), M2 (small effect) and M3 (brightening causes leakage) are consistent with a different mechanism in kidney, which these experiments do not identify.

## Limits and notes (stated, not fixed)
- M3 flagged edits: 91 of 136 Arm A edits and 91 controls, 9 of 19 Arm B edits and 9 controls missed the target mean by more than 5 HU (flagged, kept, as pre-registered). Cause: the 1-voxel feather makes the realised offset smaller than requested at large offsets (maximum miss 56 HU). Every edit and control still achieved its intended contrast sign. The realised contrast magnitudes differ by design: edit about 24 HU vs control about 97 HU in Arm A.
- **Arm B is a weak manipulation:** the bright leaking lesions have a median pre-edit contrast of 14 HU (IQR 7 to 27), so the requested equal-magnitude flip is a median offset of about 28 HU, and after feathering the realised post-edit |contrast| is a median of about 8 HU (Arm A: pre-edit 38 HU, requested offset about 75 HU, realised about 24 HU). The null therefore does not show that darkening cannot remove the leak.
- Arm selection used the unedited W2 v3 outcome (Arm A not LEAK, Arm B LEAK), as in the HCC counterfactual; Arm A baseline LEAK is 0% and Arm B 100% by construction.
- HU edits change the lesion mean only, not texture, edges or enhancement pattern.
- M2 is a single linear direction at one layer; the organ check uses a seeded 60-case subset; projection analysis uses lesion tokens defined by the token centre inside the lesion (original-grid voxel), not the M1 coverage rule.
- No multiplicity correction across M1, M2, M3.

## Named confounds
Possible in-distribution images (AbdomenCT-1K, tumour labelling there unknown); arterial/corticomedullary phase (bright cortex in the parenchyma reference); point prompts give location.
