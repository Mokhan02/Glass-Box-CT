#!/bin/bash
# W3 Stage 2, first GPU session (results/w3_models/PREREG_stage2.md), run from the repo checkout on the instance. Stops at the first failure (set -e). Paths are the persistent-filesystem layout used in Stage 1.
set -e
W=/lambda/nfs/CTGate/w3; S=$W/stage2; mkdir -p $S; A=$W/modelA; MF=$W/nnunet/results/Dataset003_Liver/nnUNetTrainerW3Seed__nnUNetPlans__3d_fullres
source $W/nnunet_env.sh                                        # nnU-Net env (venv_nnunet) for the D parts
# 1. real-case hook check, one interpreter per model (A: venv_a with monai 0.9.0; D: venv_nnunet); stop if either fails
PYTHONPATH=. $W/venv_a/bin/python scripts/w3s2_hook_check.py A $S/hook_check_A $A/repo_A $A/weights/swin_unetr.pth $A/weights/unet.pth
PYTHONPATH=. python scripts/w3s2_hook_check.py D $S/hook_check_D $MF
# 2. 3-case smoke tests of the extraction (timed)
( time PYTHONPATH=. $W/venv_a/bin/python scripts/w3s2_m1_extract.py A $S/smoke_A $A/repo_A $A/weights/swin_unetr.pth --limit 3 ) 2>&1 | tail -8
( time PYTHONPATH=. python scripts/w3s2_m1_extract.py D $S/smoke_D $MF --limit 3 ) 2>&1 | tail -8
