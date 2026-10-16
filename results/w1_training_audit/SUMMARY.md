# W1 summary: training-data contrast audit (research use only)

Rule, data and deviations: [PREREG.md](PREREG.md) (committed 9e1a515), [DEVIATIONS.md](DEVIATIONS.md) (#1 to #5), [DATASETS.md](DATASETS.md). All numbers below are copied from [w1_results.md](w1_results.md) / [w1_results.json](w1_results.json), produced by `scripts/w1_compare.py` (case-level bootstrap, 1000 resamples, seed 0, percentile 95% CIs). Bright = lesion mean HU minus liver-shell mean HU > 0, two largest lesions >= 10 mm per case; reference = liver minus all tumour minus the labelled portal vein (exact exp8 parity).

## Verdict (the pre-registered rule, applied as written): BRIGHT RARE IN TRAINING

| | bright share | lesions / cases |
|---|---|---|
| Pooled training data (MSD Task03 + CRML-CT) | 5.2% [2.9, 7.6] | 25 of 485 lesions, 307 cases |
| HCC-TACE-Seg held-out, UID route | 32.6% [21.2, 42.4] | 28 of 86 lesions, 75 cases |
| d = p_HCC - p_train | +27.4 points [+15.9, +37.7] | margin 0.10 met (CI lower > 0 and d >= 0.10); 485 training lesions (needs >= 200) |

Reading: in the public liver-tumour labels VISTA-3D drew on, tumours brighter than the surrounding liver are rare (about 1 in 19), against about 1 in 3 in the held-out HCC cohort. This is a statement about the labelled data. It does not show that the model learned the missing pattern from them (that needs W5), and it is confounded as below.

## Secondary (does not change the verdict)
- **Per dataset.** MSD Task03: 1.1% [0.0, 2.8] (2 of 184 lesions, 113 cases), d +31.5 [+20.9, +41.8]; alone it has fewer than 200 lesions, so the rule gives INCONCLUSIVE. CRML-CT: 7.6% [4.2, 11.1] (23 of 301 lesions, 194 cases), d +24.9 [+14.0, +35.7]; alone the rule gives BRIGHT RARE IN TRAINING. The two datasets point the same way.
- **Contrast quantiles (HU, 5/25/50/75/95).** MSD Task03 -74.2, -47.4, -38.0, -24.6, -12.0; CRML-CT -69.7, -42.1, -28.4, -15.2, +4.7; pooled -71.3, -45.1, -31.2, -18.4, +0.6; HCC -43.0, -21.0, -8.9, +6.2, +31.5.
- **Size strata (<30, 30-60, >60 mm), pooled training vs HCC.** d +61.5 [+26.1, +95.9] (HCC 4 of 6 bright; very small), +36.0 [+17.4, +54.5], +22.0 [+12.0, +33.3] (training 0 of 63 bright). The sign of d is the same in all three strata.
- **Sensitivity.** Bright as > +10 HU: d +19.4 [+9.1, +29.6]. CRML-CT reference also removing hepatic veins: +27.0 [+16.3, +37.5]; no vessel removal: +27.4 [+16.8, +38.1]. MSD inclusion as in DEVIATIONS #4: +27.2 [+17.2, +37.8]. All lesions >= 10 mm in training vs HCC primary: +29.3 [+19.3, +39.4].

## Confounders (named here, not only in the pre-registration)
- **Tumour type.** CRML-CT is colorectal liver metastases, usually hypodense in portal-venous phase; MSD Task03 is a mix of primary and secondary tumours; the held-out cohort is HCC, which enhances in arterial phase and washes out later. A low bright share in training is partly biology, not only a labelling or selection effect.
- **Acquisition phase.** The training sets are mostly portal-venous (CRML-CT stated; Task03 not verified); HCC-TACE-Seg mixes acquisitions and its phase cannot be labelled (the aorta/portal heuristic agreed with reference contrast no better than chance, kappa 0.04). The gap therefore cannot be split into labels versus phase mix. W4 and W5 address that.

## Other limits
- Which cases fell in VISTA-3D's 64% training split is not public; all labelled cases were audited. The MAISI synthetic-tumour stage is not covered.
- The HCC reference is the 75 UID-route cases (86 lesions), a small sample; the 89-case clean cohort is a planned sensitivity analysis. The HCC reproduction passed: 86 of 86 lesions, max difference 0.0000 HU ([hcc_repro.log](hcc_repro.log)).
- MSD Task03: 115 of 131 cases entered (13 tumour-free, 3 failed the surface guard: liver_39, liver_71, liver_74); 113 contributed lesions. CRML-CT: 197 of 197 entered, 194 contributed lesions ([inventories](msd_inventory.csv)).
- Mean HU of a lesion mixes necrosis and calcification; HU compared raw across scanners.
