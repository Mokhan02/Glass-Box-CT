"""W5 Amendment 4 smoke trainers: the real BRIGHT-25 and D0-seed2 trainers with num_epochs = 2 (own result folders, nothing mixes with the real arms). The transform's core call is logged to $W5_EDIT_LOG by importing nnUNetTrainerW5Smoke
(a side effect of that module); the D0-seed2 trainer has no augmentation, so its edit log must stay empty. Installed next to the other W5 trainer files as `nnUNetTrainerW5SmokeA4.py`."""
import torch
from nnunetv2.training.nnUNetTrainer import nnUNetTrainerW5Smoke  # noqa: F401  (side effect: logs every shift_tumour_contrast call to $W5_EDIT_LOG)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5 import nnUNetTrainerW5Bright25, nnUNetTrainerW5D0Seed2


class nnUNetTrainerW5SmokeBright25(nnUNetTrainerW5Bright25):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):      # explicit names: nnU-Net records constructor arguments from the signature
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.print_to_log_file("W5 SMOKE (Bright25): 2 epochs; edit log $W5_EDIT_LOG")


class nnUNetTrainerW5SmokeD0Seed2(nnUNetTrainerW5D0Seed2):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.print_to_log_file("W5 SMOKE (D0Seed2): 2 epochs; no augmentation, the edit log must stay empty")
