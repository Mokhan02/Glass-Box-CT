# W2-M route check write-up: is M2's null due to the zero-shot route rather than the organ?

Numbers: [route/route_results.md](route/route_results.md) / [route_results.json](route/route_results.json), from the committed scripts, unchanged. Rule: [PREREG_route.md](PREREG_route.md). Notes: [DEVIATIONS.md](DEVIATIONS.md) #5.

## Verdict (pre-registered rule): ORGAN DIFFERS
Same route (label 133 + point label 1), same mid-encoder hook, same margins; only the organ is changed back to HCC (76 main-set lesions, 68 cases, cross-fitted HCC mech-B erasers).
- **Primary dD (HCC 133/1, LEACE, change in dice_full): -0.104 [-0.139, -0.071].** The M2 USED criteria are met: (1) at most -0.05 with the CI below zero; (2) LEACE minus random -0.105 [-0.143, -0.071] and LEACE minus HU -0.108 [-0.148, -0.070], both CI uppers below -0.02; (3) the liver organ Dice is untouched (-0.000 [-0.001, +0.000], 30-case subset). The random and HU controls change Dice by +0.001 [-0.001, +0.004] and +0.004 [-0.020, +0.031].
- **ROUTE EXPLAINS M2 is excluded:** it needed the CI lower bound above -0.10; the observed lower bound is -0.139.
- **Positive control passes:** class 26 on the automatic route (no point), the same erasers: 10 mm Dice change -0.350 [-0.408, -0.298] on the 63 of 76 lesions the unhooked run detects (all 76: -0.295 [-0.351, -0.244]); mech B reported -0.33 [-0.38, -0.29]. The hook and erasers do what they did in mech B, so the zero-shot result is readable.
- **Secondary arm HCC 26/3:** also USED (dice_full change -0.125 [-0.158, -0.094]; 10 mm Dice -0.144 [-0.180, -0.109]); organ Dice unchanged.
- Failure modes under HCC LEACE (133/1): UNDER 29% -> 49%, LEAK 12% -> 8%, OK 59% -> 43% (26/3: UNDER 22% -> 55%).

## Reading
The HCC dependence on this mid-encoder direction is present on the zero-shot point route (about -0.10 in Dice, no more than a few hundredths beyond random or HU directions would give, organ untouched) while, with the same hook and the same decision margins, the kidney arm showed -0.015 (kidney-native direction; M2) and -0.016 (HCC direction applied to kidney; M2 secondary). So M2's null is not explained by the route: **kidney differs from liver even on the same route.** The direction of the effect also differs: erasing it in HCC mainly increases under-segmentation (UNDER up about 20 to 33 points), whereas in kidney it slightly increased leakage (LEAK +6 points). The route does shrink the effect relative to the automatic route (-0.10 on the point route vs about -0.33 to -0.35 automatic), so route matters in size but not in whether an effect exists.

## Limits and notes (stated, not fixed)
- Different lesion populations: HCC and kidney differ in tumour type, contrast phase mix, size and baseline segmentation quality (W2 v3 mean Dice: HCC 133/1 dark 0.693, bright 0.448; kidney 133/1 dark 0.517, bright 0.414), so a Dice drop is not like-for-like in absolute terms; the comparison is about whether each organ's own direction matters under each route.
- The erasers are cross-fitted HCC mech-B directions (fitted on training folds, applied to held-out folds); the kidney-native eraser of M2 was fitted on kidney training folds; both are single linear directions at one layer.
- The positive control reproduces mech B's route, not its exact numbers (mech B prompted classes 1 and 26 together and read out inside the predicted-liver region).
- 76 lesions in 68 cases (23 bright); the CIs are wider than M2's but the effect is far outside them.
- HCC-TACE-Seg mixes acquisitions and its phase cannot be labelled; the kidney confounds (possible training overlap via AbdomenCT-1K, arterial phase) do not apply to HCC.
- No multiplicity correction across M1 to M3 and this check.
