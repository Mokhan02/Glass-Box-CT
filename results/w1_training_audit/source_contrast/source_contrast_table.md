| Dataset | Phase | Cases | Lesions (2 largest per case) | Bright (c > 0) | Dark (c <= 0) | Bright share [95% CI] | c quantiles 5/25/50/75/95 (HU) | All lesions >= 10 mm: bright / total |
|---|---|---|---|---|---|---|---|---|
| MSD Task03 (training source) | portal-venous in most cases (LiTS, several institutions); phase not labelled per case | 113 | 184 | 2 | 182 | 1.1% [0.0, 2.8] | -74, -47, -38, -25, -12 | 3 / 574 |
| CRML-CT (training source) | portal-venous (preoperative contrast-enhanced CT) | 194 | 301 | 23 | 278 | 7.6% [4.4, 11.2] | -70, -42, -28, -15, +5 | 29 / 408 |
| HCC-TACE-Seg (held-out reference) | mixed acquisitions; phase not labelled (tags stripped) | 75 | 86 | 28 | 58 | 32.6% [22.1, 42.4] | -43, -21, -9, +6, +31 | n/a (primary unit only) |
| WAW-TACE, contrast on the phase the mask was drawn on (held-out reference) | arterial (phase 1) or portal-venous (phase 2), whichever the annotators drew on; 181 of 276 primary lesions drawn on arterial | 199 | 276 | 201 | 75 | 72.8% [66.9, 78.6] | -23, -0, +12, +19, +35 | 245 / 329 |
| WAW-TACE arterial series, all 297 paired tumours | arterial | 181 | 297 | 242 | 55 | 81.5% [75.4, 86.9] | -11, +3, +12, +19, +33 | n/a (primary unit only) |
| WAW-TACE portal-venous series, same 297 tumours | portal-venous | 181 | 297 | 166 | 131 | 55.9% [49.3, 62.4] | -26, -8, +2, +11, +24 | n/a (primary unit only) |

WAW-TACE primary lesions by the phase the mask was drawn on: arterial-drawn: 154/181 bright (85.1%); portal-drawn: 47/95 bright (49.5%)
