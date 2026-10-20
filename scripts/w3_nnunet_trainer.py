"""W3 model D trainer: nnU-Net v2 `nnUNetTrainer` (default 1000 epochs x 250 iterations, default plans) with ONE change, a fixed seed set before the trainer is constructed (Python, NumPy, torch, CUDA).
It is installed by copying this file to `<venv_nnunet>/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerW3Seed.py` (nnU-Net finds trainers by class name inside that package); the file's hash is in results/w3_models/code_hashes.txt.
NOTE: seeding does not make training bitwise deterministic here: torch.compile, cuDNN autotuning, non-deterministic CUDA kernels and the multi-process data augmentation workers are not reproducible bit for bit, and a run resumed from a checkpoint continues from a re-seeded state."""
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

SEED = 20261020


class nnUNetTrainerW3Seed(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        super().__init__(plans, configuration, fold, dataset_json, device)
        assert self.num_epochs == 1000, "W3 schedule is 1000 epochs (PREREG_stage1 schedule rule, Gate 0 benchmark)"
        self.print_to_log_file(f"W3: seed {SEED} set before construction; epochs {self.num_epochs}; torch.compile is not bitwise deterministic")
