# W4b / W6 analysis (PREREGs applied as written)

## 1. W4b PRIMARY: theta(BRIGHT) - theta(SHAM) (one-sided alpha 0.025)

{"d": -0.27711122703297975, "lo": -0.4181084813781723, "hi": -0.1348419264896157, "boot_ok": 2000, "label": "REDUCES THE PENALTY", "n_tumours": 297}

## 2. W4b secondaries

- S1_BRIGHT_minus_PLAIN: {"d": -0.24059583840491508, "lo": -0.3842978804424996, "hi": -0.08978705325682082, "boot_ok": 2000, "label": "REDUCES THE PENALTY", "n_tumours": 297}
- S2_DEC_minus_SHAM: {"d": -0.23277111227032976, "lo": -0.3754310876934073, "hi": -0.07894729150517965, "boot_ok": 2000, "label": "REDUCES THE PENALTY", "n_tumours": 297}
- S3_BRIGHT_NO_PENALTY: "NO PENALTY"

## 3. Per-model W4b labels (ALREADY SEEN = analysed in W4 before W4b)

- BASELINE (ALREADY SEEN): theta +0.159 [+0.036, +0.280] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): INCONCLUSIVE; F1 81, NF 211
- A (ALREADY SEEN): theta +0.314 [+0.188, +0.439] -> BRIGHT PENALTY; NO PENALTY test (margin 0.25): INCONCLUSIVE; F1 81, NF 211
- D0 (ALREADY SEEN): theta +0.352 [+0.222, +0.483] -> BRIGHT PENALTY; NO PENALTY test (margin 0.25): INCONCLUSIVE; F1 81, NF 211
- B50 (ALREADY SEEN): theta -0.060 [-0.184, +0.056] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- BRIGHT: theta -0.032 [-0.164, +0.100] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- SHAM: theta +0.245 [+0.121, +0.379] -> BRIGHT PENALTY; NO PENALTY test (margin 0.25): INCONCLUSIVE; F1 81, NF 211
- PLAIN: theta +0.209 [+0.089, +0.338] -> BRIGHT PENALTY; NO PENALTY test (margin 0.25): INCONCLUSIVE; F1 81, NF 211
- BRIGHT-DEC: theta +0.013 [-0.127, +0.152] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- BRIGHT-s2: theta -0.059 [-0.191, +0.090] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- TS: theta -0.034 [-0.150, +0.080] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- W6b-FULL-L143: theta +0.031 [-0.119, +0.189] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- W6b-FULL-L573: theta -0.036 [-0.179, +0.119] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211
- W6b-DEC-L143: theta +0.037 [-0.107, +0.201] -> INCONCLUSIVE; NO PENALTY test (margin 0.25): NO PENALTY; F1 81, NF 211

## 4. W6 (c)-type tests on the arterial series

- BRIGHT_vs_SHAM (PRIMARY, W6): CONFIRMED (detection and Dice)
- BRIGHT_vs_PLAIN: CONFIRMED (detection and Dice)
- BRIGHT_vs_BASELINE (descriptive, confounded with fine-tuning): CONFIRMED (detection and Dice)
- BRIGHT-s2_vs_SHAM (replication): CONFIRMED (detection and Dice)
- DEC_vs_SHAM: CONFIRMED (detection and Dice)
- Q2_decoder_only_ratio: None
- seed_difference_of_gaps (BRIGHT vs BRIGHT-s2, arterial): None

## 7. Sensitivities: verdicts that change against the primary

- dilation_2mm_propagated: no verdict changes
- top2_per_series: no verdict changes
- excl_value2_patients: no verdict changes
- incl_Q: no verdict changes
- excl_alignment_gate_failures: no verdict changes
- min_diam_15mm: no verdict changes
- liver_eroded_3mm: w4b/S3_BRIGHT_NO_PENALTY: NO PENALTY -> INCONCLUSIVE; w4b/model/PLAIN: BRIGHT PENALTY -> INCONCLUSIVE
- shell_median: no verdict changes
