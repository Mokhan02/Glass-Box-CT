# C. Steering (sufficiency) at the mid-encoder and bottleneck (pre-registered; results/mech/PREREG.md section C)

Probe direction added to every feature vector in the ROI cells (predicted liver + 10 mm, no ground truth at test time), alpha chosen per fold on 20 training cases, compared with a threshold shift matched on healthy-liver false-positive volume, random directions and the mean-difference direction. 89 cases, 5-fold patient-level CV, held-out folds only. Original readout: Dice 0.555, detection 77.2%, SLD median error -19.2%.

| layer | arm | Dice [95% CI] | detection | SLD median error | >= 20% over-estimated | FP volume (cm3) |
|---|---|---|---|---|---|---|
| mid | threshold (matched FP) | 0.579 [0.512, 0.644] | 79.2% | -16.5% | 0.0% | 1.67 |
| mid | **probe steering** | 0.573 [0.504, 0.640] | 78.2% | -15.4% | 0.0% | 1.64 |
| mid | random direction | 0.554 [0.486, 0.621] | 77.2% | -20.9% | 0.0% | 1.31 |
| mid | mean-difference direction | 0.576 [0.508, 0.641] | 80.2% | -15.0% | 1.1% | 2.32 |
| bottleneck | threshold (matched FP) | 0.578 [0.512, 0.642] | 80.2% | -16.4% | 0.0% | 1.65 |
| bottleneck | **probe steering** | 0.575 [0.508, 0.641] | 78.2% | -17.5% | 0.0% | 1.66 |
| bottleneck | random direction | 0.554 [0.484, 0.621] | 75.2% | -22.4% | 0.0% | 1.28 |
| bottleneck | mean-difference direction | 0.622 [0.560, 0.681] | 85.1% | -10.2% | 3.4% | 3.60 |

Paired per-case Dice differences, probe steering minus: threshold -0.006 [-0.013, +0.001] (mid) and -0.003 [-0.009, +0.002] (bottleneck); random direction +0.020 [+0.009, +0.030] and +0.021 [+0.014, +0.029]; mean-difference -0.003 [-0.015, +0.008] and -0.048 [-0.067, -0.030].

**Pre-registered verdicts: mid-encoder EQUIVALENT TO THRESHOLD; bottleneck EQUIVALENT TO THRESHOLD.** Probe steering beats random directions (both layers) but not the matched threshold shift. Bright vs dark (descriptive): Dice at mid with probe steering 0.337 / 0.699 against the original 0.31 / 0.68 (bright / dark; `C_table.csv`).

Plain-language summary: pushing the mid-level or bottleneck features along the tumour-vs-liver direction inside the liver region recovers some missed tumour (Dice 0.555 to 0.575, detection 77% to 78%), clearly more than pushing along a random direction, but no more than simply lowering the output threshold by the amount that adds the same amount of false-positive liver. So at the strengths tested the feature is sufficient to raise detection but behaves like a threshold shift, not like a better way to find the tumour. Limits that matter: (1) the chosen alpha was the largest grid value (1.5) in all 5 folds at both layers, so the optimum is at the edge of the pre-registered grid and stronger steering was not tested; (2) matching on false-positive volume makes the comparator a small threshold shift (about -0.5 logit), well short of the Dice-optimal shift found earlier (about -3), so both arms are weak operating points; (3) the mean-difference direction at the bottleneck gives the highest Dice (0.622) but with about twice the false-positive volume and 3.4% over-estimation, so it is not comparable to the matched threshold arm and no claim is made from it (exploratory observation only).
