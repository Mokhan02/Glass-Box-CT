# HCC reproduction and solver checks (W2-M, conditions 1 and 2 of the researcher's acceptance of DEVIATIONS #1)

Environment of these re-runs: scikit-learn 1.7.2, numpy 2.2.6 ([environment_lock.txt](../../environment_lock.txt)). The stored HCC tables come from an earlier environment that no longer exists. Re-runs use `scripts/voxel_probe_analyze.py` unchanged on the stored HCC tokens (`--skip-recovery`); comparison by `scripts/w2m_hcc_compare.py`, quoted numbers by `scripts/w2m_hcc_quoted.py`.

| comparison | differing cells | max abs diff | max abs diff at mid | verdict same | decision booleans same | trained-vs-control comparisons changed |
|---|---|---|---|---|---|---|
| determinism: run 2 vs run 1 | 0 / 1086 | 0.0000 | 0.0000 | True (LOST LATE) | True | 0 of 15 |
| primary_default_run2 | 228 / 1086 | 0.0070 | 0.0023 | True (LOST LATE) | True | 0 of 15 |
| primary_corrected | 278 / 1224 | 0.0070 | 0.0023 | True (LOST LATE) | True | 0 of 30 |
| single_acq | 203 / 1086 | 0.0040 | 0.0040 | True (LOST LATE) | True | 0 of 15 |
| single_acq_corrected | 221 / 1224 | 0.0040 | 0.0040 | True (LOST LATE) | True | 0 of 30 |
| uid | 144 / 930 | 0.0012 | 0.0004 | True (LOST LATE) | True | 0 of 12 |
| uid_corrected | 185 / 1044 | 0.0009 | 0.0005 | True (LOST LATE) | True | 0 of 24 |
| tight solver (tol 1e-8, max_iter 20000) vs stored | 529 / 1086 | 0.0210 | 0.0017 | True (LOST LATE) | True | 0 of 15 |
| tight vs default solver (same environment) | 526 / 1086 | 0.0210 | 0.0014 | True (LOST LATE) | True | 0 of 15 |

## Quoted numbers (results/SUMMARY.md section 2 / README): 27 cells compared at 3 decimals; 7 differ, each by 0.001:

- all-103, bottleneck, trained activations: stored (0.825, 0.795, 0.869), re-run (0.825, 0.796, 0.869)
- all-103, decoder_F, trained raw-HU: stored (0.11, 0.048, 0.141), re-run (0.109, 0.048, 0.141)
- single-acquisition, mid, trained raw-HU: stored (0.084, 0.041, 0.115), re-run (0.083, 0.041, 0.115)
- single-acquisition, bottleneck, trained raw-HU: stored (0.19, 0.075, 0.247), re-run (0.19, 0.074, 0.247)
- UID-route, bottleneck, trained raw-HU: stored (0.19, 0.061, 0.253), re-run (0.191, 0.061, 0.254)
- UID-route, decoder_F, trained raw-HU: stored (0.112, 0.044, 0.167), re-run (0.113, 0.045, 0.167)
- UID-route, decoder_F, random-init activations (corrected control): stored (0.127, 0.044, 0.161), re-run (0.127, 0.044, 0.162)

## Reading
- Determinism: run 2 reproduces run 1 exactly (0 of 1086 cells differ), so the differences from the stored tables are environmental, not random.
- Category: no verdict (LOST LATE in all six result sets), decision boolean or trained-vs-control comparison changes in any re-run; the primary row (trained, mid, probe A, pooled) is exact.
- Solver tolerance matters in the high-dimensional layers: with tol 1e-8 and max_iter 20000 the HCC table moves by up to 0.021 (bottleneck / small subsets) but by at most 0.0017 at the mid layer; the stored default-tolerance results are not at the converged solution there. M1 uses the mid layer with the HCC specification (default tolerance), as committed.
