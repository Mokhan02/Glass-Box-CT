# Flag-gated recalibration (pre-registered, results/trust/PREREG.md section B)

89 clean-cohort cases (89 with a measurable target); 5-fold patient-level CV; the flag is refitted on training folds; 49 cases were gated to the recalibrated threshold.

| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |
|---|---|---|---|---|---|
| default | -19.2% | 34.1% | 0.0% | 49.4% | 0.555 |
| recalibrated | -0.1% | 21.9% | 12.4% | 31.5% | 0.640 |
| gated | -7.7% | 20.6% | 3.4% | 29.2% | 0.647 |

Gated minus always-recalibrated, over-estimation indicator: -0.090 (95% CI -0.157 to -0.034).
Gated minus always-default, absolute sum-of-diameters error: -0.135 (95% CI -0.195 to -0.080).

**Decision (pre-registered rule): SHIPPED: gated policy beats both.**
