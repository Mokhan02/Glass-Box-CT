# W5 Amendment 4: dose-response of the bright-versus-dark gap

**Overall label: DOSE-RESPONSE (Dice only)** (74 cases; gaps = dark minus bright)

| arm | p | detection gap | Dice gap |
|---|---|---|---|
| D0 | 0.0 | +0.645 | +0.419 |
| B25 | 0.25 | +0.107 | +0.201 |
| B50 | 0.5 | +0.090 | +0.185 |
| B50S2 | 0.5 | +0.125 | +0.193 |

- **detection**: g0 +0.645, g25 +0.107, g5 +0.107 (separation 0.538); C_low = g25 - g0 -0.538 [-0.746, -0.330]; C_high = g25 - g5 -0.000 [-0.045, +0.050] -> **NONE**
- **Dice**: g0 +0.419, g25 +0.201, g5 +0.189 (separation 0.230); C_low = g25 - g0 -0.218 [-0.358, -0.089]; C_high = g25 - g5 +0.012 [-0.020, +0.047] -> **DOSE-RESPONSE**, curve saturating

Descriptive (no verdict):
- detection: slope per unit p -0.978 [-1.343, -0.619]; B25 residual from the leave-out line -0.269 [-0.378, -0.149]; baseline seed variance unavailable (D0-seed2 not run); |gap(B50) - gap(B50S2)| = 0.036
- Dice: slope per unit p -0.422 [-0.652, -0.208]; B25 residual from the leave-out line -0.103 [-0.179, -0.029]; baseline seed variance unavailable (D0-seed2 not run); |gap(B50) - gap(B50S2)| = 0.008
