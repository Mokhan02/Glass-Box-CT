# Bridge between B and D (EXPLORATORY, post hoc; results/mech/DEVIATIONS.md #5)

Same tumours, dark original vs the bright edit (D's 62 Arm A pairs, 58 cases, all complete). Score t = amount of the LEACE-erased tumour-vs-liver direction present in the mid-encoder cells inside the lesion, using the fold's held-out eraser (units: multiples of the erased direction).

| set | tumours | mean projection, dark | bright minus dark | sign test (bright lower) | projection drop vs coverage drop |
|---|---|---|---|---|---|
| all complete | 62 (58 cases) | 4.54 | **-0.93 [-1.22, -0.66]** | 55 of 62, p = 2e-10 | Pearson 0.71 [0.59, 0.84]; Spearman 0.74 [0.58, 0.85] |
| eligible (coverage drop >= 0.20) | 32 (30 cases) | 4.51 | **-1.53 [-1.98, -1.12]** | 30 of 32, p = 2e-7 | Pearson 0.54 [0.32, 0.77]; Spearman 0.62 [0.33, 0.81] |

By the rule fixed beforehand (paired difference CI below zero and Spearman CI above zero on the eligible set): **the projection tracks the brightness effect.**

Plain-language summary: making the same tumour brighter lowers how much of the tumour-vs-liver direction (the one B shows the model uses) is present in the mid-encoder features inside that tumour: in 55 of 62 tumours, and by about a third of its dark-version value on the eligible tumours (1.5 of 4.5). Tumours whose projection drops more also lose more coverage (rank correlation 0.6 to 0.7). So the chain is: brightness lowers the tumour-vs-liver feature in the mid-encoder (this bridge), the feature is necessary for detection (B), and restoring the dark mid-encoder features inside the tumour restores detection (D). Limits: exploratory; correlational across tumours (the causal tests are B and D); one linear functional per fold; the edit is synthetic.
