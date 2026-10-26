# W5 results (SINGLE-SEED: one training run per arm; training-run variance is not measured and the patient bootstrap does not cover it)

- D0: detection gap +0.645, Dice gap +0.419, mean Dice 0.390, median signed SLD error -0.343
- B50: detection gap +0.090, Dice gap +0.185, mean Dice 0.637, median signed SLD error -0.099
- S50: detection gap +0.520, Dice gap +0.431, mean Dice 0.437, median signed SLD error -0.255
- B50_minus_D0: detection -0.556 [-0.774, -0.343], Dice -0.234 [-0.367, -0.108]
- B50_minus_S50: detection -0.430 [-0.656, -0.202], Dice -0.246 [-0.370, -0.120]
- verdicts: {'D0_det': 'CLOSES', 'D0_dice': 'CLOSES', 'S50_det': 'CLOSES', 'S50_dice': 'CLOSES'}; prediction: CONFIRMED (detection and Dice)
