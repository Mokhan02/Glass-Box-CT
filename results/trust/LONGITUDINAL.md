# Two-timepoint check: bias model and simulated scenarios (SIMULATED)

SIMULATED: generated from a fitted bias model, not real patients.

Modelling assumption: a lesion's detection draw carries over between visits with probability 0.5 (the `persistence` parameter; 0 = independent misses, which exaggerates category flips; 1 = identical). It is a stated assumption, not a fitted quantity, and is adjustable in the app.

Bias model fitted on 101 real per-lesion measurements (36 in bright-tumour cases). Per threshold: detection = logistic(1, log true size, bright); detected diameter = a + b * true + c * bright + noise.

| threshold | detected lesions | median error of detected (mm) | detection coefficients | size model (a, b, c) | noise sd (mm) |
|---|---|---|---|---|---|
| default | 78 | -10.9 | -4.54, +1.63, -1.78 | +6.99, +0.81, -11.98 | 16.0 |
| recalibrated | 90 | +0.6 | -5.06, +2.03, -1.57 | +18.46, +0.83, -13.34 | 19.3 |

Across 304 simulated scenarios (true change -50% to +40%, four baselines, four appearance patterns) the default and recalibrated measurements give different RECIST categories in at least 25% of draws in 145 (48%). The default threshold is wrong more often than the recalibrated one in 94% of scenarios and less often in 6%.

Illustrative scenarios shown in the app:

- Simulated patient A: true PR, bright to bright: baseline [60.0, 30.0] mm, true change -35%; default most likely CR, recalibrated most likely PR (differ in 45% of draws).
- Simulated patient B: true SD, bright to bright: baseline [60.0, 30.0] mm, true change -30%; default most likely CR, recalibrated most likely PR (differ in 43% of draws).
- Simulated patient C: true PD, dark to bright (phase change): baseline [60.0, 30.0] mm, true change +20%; default most likely PR, recalibrated most likely SD (differ in 41% of draws).
