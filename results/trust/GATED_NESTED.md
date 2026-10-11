# Flag-gated recalibration: nested-CV SENSITIVITY analysis (not pre-registered)

The recalibrated threshold (grid -1.0 to -4.0 logit) and the flag's sensitivity target (0.80 to 0.95) are chosen inside each outer training fold only, by an inner 4-fold cross-validation of the gated policy's mean absolute error; always-recalibrated gets its own threshold chosen inside the training fold. 95% patient-bootstrap CIs in brackets.

89 cases with a measurable target.

| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |
|---|---|---|---|---|---|
| default | -19.2% [-28.9, -14.3] | 34.1% [26.7, 41.8] | 0.0% [0.0, 0.0] | 49.4% | 0.555 [0.486, 0.622] |
| recalibrated | -0.1% [-6.3, +2.7] | 21.9% [16.3, 28.5] | 12.4% [5.6, 20.2] | 31.5% | 0.640 [0.583, 0.694] |
| gated | -4.5% [-6.2, -2.0] | 19.8% [15.2, 25.2] | 7.9% [3.4, 14.6] | 31.5% | 0.645 [0.590, 0.698] |

Gated minus always-recalibrated, over-estimation indicator: -0.045 (95% CI -0.124 to +0.034). Gated minus always-default, absolute error: -0.143 (95% CI -0.202 to -0.085). The pre-registered rule applied to this analysis would NOT ship the gated policy.

Parameters chosen per outer fold:

- fold 0: gated threshold -3.5, flag sensitivity 0.9 (11 held-out cases gated); always-recalibrated threshold -3.0
- fold 1: gated threshold -3.5, flag sensitivity 0.9 (10 held-out cases gated); always-recalibrated threshold -3.0
- fold 2: gated threshold -3.5, flag sensitivity 0.9 (8 held-out cases gated); always-recalibrated threshold -3.0
- fold 3: gated threshold -3.5, flag sensitivity 0.9 (13 held-out cases gated); always-recalibrated threshold -3.0
- fold 4: gated threshold -3.5, flag sensitivity 0.95 (8 held-out cases gated); always-recalibrated threshold -3.0

**Quoted in the app and README: the nested sensitivity analysis.** Gated over-estimation 7.9% here vs 3.4% pre-registered; gated mean absolute error 19.8% vs 20.6%; this analysis does NOT meet the pre-registered shipping rule (see results/trust/DEVIATIONS.md #1 for why the quoting rule was changed after both results were seen).
