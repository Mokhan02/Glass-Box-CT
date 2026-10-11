# D. Counterfactual patching (pre-registered; results/mech/PREREG.md section D)

Arm A, dark tumours made bright (62 lesions run, all complete; **32 eligible** with a coverage drop >= 0.20 after the edit). BRIGHT run with DARK activations patched in per layer and region; restored fraction = share of the dark-to-bright coverage drop recovered, pooled over eligible lesions, patient-bootstrap 95% CI. The unpatched dark and bright runs reproduced the earlier counterfactual coverages on every checked lesion.

| layer | R1 tumour + 10 mm | R2 shell only | R3 far context | R1 from an unrelated case |
|---|---|---|---|---|
| mid-encoder | **0.79 [0.71, 0.85]** | 0.06 [0.00, 0.14] | 0.03 [0.02, 0.05] | -0.42 [-0.65, -0.25] |
| bottleneck | **0.66 [0.55, 0.76]** | 0.12 [0.06, 0.20] (see DEVIATIONS 3) | 0.10 [0.06, 0.17] | -0.40 [-0.64, -0.23] |
| decoder_F | 1.00 [1.00, 1.00] (by construction) | 0.01 [0.01, 0.01] | 0.00 [0.00, 0.00] | -0.40 [-0.62, -0.22] |

Earliest layer restoring >= 50% (pre-registered readout): **mid-encoder for the tumour region**; shell-only and far-context patches never reach 50%. The unrelated-case control restores less than R1 at every layer (it makes coverage worse than the bright run). Core-logit outcome is in `D_table.csv`.

Plain-language summary: the loss of the tumour when it is made bright is already present in the encoder's mid-level features, and it lives in the tumour itself. Putting the dark scan's mid-level features back inside the tumour and a 10 mm margin restores about four fifths of the lost coverage; doing it only in the surrounding shell restores almost nothing, and so does doing it everywhere else. Patching at the decoder output restores everything inside the region, but that is by construction (it replaces what the logit is read from). Patching features from a different patient's scan at the same location makes things worse, so the effect is specific to this tumour's own features. Limits: mid is the earliest layer tested, so the effect arises at or before the mid-encoder, not necessarily there; the shell-only result at the bottleneck is largely uninformative (DEVIATIONS 3); and the edit is synthetic, so this is evidence about the model, not about patients.
