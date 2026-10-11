# Contrast-phase heuristic on HCC-TACE-Seg (descriptive; rule and thresholds pre-registered in PREREG.md section A)

89 clean-cohort cases; phase estimated from the model's own aorta and portal-vein masks. A research heuristic: no threshold was tuned, nothing is gated on this.

Phase estimates: arterial-like 21, venous-like 65, unenhanced 3, indeterminate 0.

**Arterial-like vs reference contrast > 0** (89 cases with a definite estimate): agreement 60% (95% CI 49% to 70%), kappa 0.04 (CI -0.16 to 0.24); both 8, arterial-like only 13, bright only 23, neither 45.

**Largest reference lesion missed (covered < 10% by the default prediction):**

| group | cases | miss rate (95% CI) |
|---|---|---|
| blind-spot warning fires | 25 | 28% (11% to 46%) |
| warning quiet | 64 | 17% (8% to 28%) |
| phase arterial-like | 21 | 33% (13% to 54%) |
| phase not arterial-like | 68 | 16% (8% to 26%) |

Difference in miss rate, warning fires minus quiet: +11 points (95% CI -9 to +31).

Limits: one disease, one dataset; agreement is with a physical proxy (contrast sign), not true phase labels, which do not exist here.
