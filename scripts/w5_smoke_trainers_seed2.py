"""W5 Amendment 2 smoke trainer: `nnUNetTrainerW5SmokeBright50Seed2` = the real Seed2 trainer with num_epochs = 2 and the transform's core call logged to $W5_EDIT_LOG (the logging wrapper is installed by importing `nnUNetTrainerW5Smoke`, committed earlier). Explicit constructor arguments (nnU-Net records them by name).
Installed as `nnUNetTrainerW5SmokeSeed2.py` next to the other W5 trainer files; results land in their own folder."""
import torch
from nnunetv2.training.nnUNetTrainer import nnUNetTrainerW5Smoke  # noqa: F401  (side effect: logs every shift_tumour_contrast call to $W5_EDIT_LOG)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5Seed2 import nnUNetTrainerW5Bright50Seed2


class nnUNetTrainerW5SmokeBright50Seed2(nnUNetTrainerW5Bright50Seed2):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.print_to_log_file("W5 SMOKE (Bright50Seed2): 2 epochs; edit log $W5_EDIT_LOG")
