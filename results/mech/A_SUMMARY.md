# A. Direct logit attribution (pre-registered; results/mech/PREREG.md section A)

Clean cohort, 89 cases (82 with both detected and missed tumour voxels), voxel-level decoder_F inside the predicted liver dilated 10 mm. The class-26 logit is the sum of 48 per-channel contributions w_j F_j (reconstruction error < 0.011 logit, fp16 storage).

| quantity | value |
|---|---|
| mean logit gap, detected minus missed tumour voxels | 5.55 |
| share of that gap carried by the 8 channels chosen on training folds (held-out) | **0.77 [0.76, 0.78]** -> pre-registered verdict **CLEARLY CONCENTRATED** (rule: >= 0.50) |
| smallest k reaching 50% (all data, optimistic) | 5 channels |
| top 8 channels (all data) | 36, 20, 10, 22, 18, 30, 17, 7 |
| bright-minus-dark logit gap on all tumour voxels (sum over channels) | -3.53 |
| share of the bright-minus-dark gap in the same top 8 | 0.76 [0.71, 0.83] (channels re-selected inside each bootstrap) |
| Spearman across 48 channels, missed-vs-detected gap vs bright-vs-dark gap | -0.95 [-0.98, -0.85] |
| pre-registered "same channels" verdict | **false by the literal rule** (needs a positive correlation; see DEVIATIONS 1) |
| cosine(w, decoder_F probe direction) | 0.35 (per fold 0.33 to 0.40) |

Plain-language summary: whether the model calls a voxel tumour is carried by a handful of the 48 decoder channels: eight of them account for about three quarters of the difference between voxels the model finds and voxels it misses, and this holds on cases the choice of channels never saw. The same eight channels also carry most of the difference between bright and dark tumours, and they do so in the matching direction: bright tumours are lower exactly where missed voxels are lower (rank correlation -0.95 because the two gaps have opposite signs). The pre-registered rule wrote the sign the other way round, so its literal verdict is "no"; the sign-aware reading is post hoc and exploratory. The probe direction and the readout vector agree only moderately (cosine 0.35), so the head is not simply reading the direction the probe finds. This is attribution, not causation; B to D test whether these features are used and sufficient.
