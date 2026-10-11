# Gate for the readout re-fit

Rule fixed in `scripts/voxel_probe_gate.py` before the clean-cohort and corrected-control runs finished.

## (a) governing verdict from the clean cohorts

| cohort | verdict | R_enc [95% CI] | R_hu [95% CI] | median interior-FN logit |
|---|---|---|---|---|
| single-acquisition (81) | LOST LATE | 0.604 [0.571, 0.676] | 0.083 [0.041, 0.115] | -2.53 |
| UID route (75) | LOST LATE | 0.627 [0.587, 0.713] | 0.112 [0.045, 0.171] | -2.40 |

Governing verdict: **LOST LATE** -> condition (a): **PASS**

## (b) trained mid-encoder vs corrected random-init control (probe A, interior-FN detection)

| cohort | trained mid [95% CI] | random-init mid [95% CI] | trained CI lower > random CI upper? |
|---|---|---|---|
| single-acquisition (81) | 0.604 [0.571, 0.676] | 0.078 [0.060, 0.104] | yes |
| UID route (75) | 0.627 [0.587, 0.714] | 0.075 [0.057, 0.101] | yes |

Condition (b): **PASS** (must hold in both clean cohorts)

## GATE: **PASSED**
