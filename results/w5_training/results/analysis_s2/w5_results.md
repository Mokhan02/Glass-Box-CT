# W5 results (SINGLE-SEED: one training run per arm; training-run variance is not measured and the patient bootstrap does not cover it)

- D0: detection gap +0.645, Dice gap +0.419, mean Dice 0.390, median signed SLD error -0.343
- B50: detection gap +0.125, Dice gap +0.193, mean Dice 0.643, median signed SLD error -0.091
- S50: detection gap +0.520, Dice gap +0.431, mean Dice 0.437, median signed SLD error -0.255
- B50_minus_D0: detection -0.520 [-0.738, -0.310], Dice -0.226 [-0.353, -0.106]
- B50_minus_S50: detection -0.395 [-0.618, -0.177], Dice -0.238 [-0.359, -0.114]
- verdicts: {'D0_det': 'CLOSES', 'D0_dice': 'CLOSES', 'S50_det': 'CLOSES', 'S50_dice': 'CLOSES'}; prediction: CONFIRMED (detection and Dice)
