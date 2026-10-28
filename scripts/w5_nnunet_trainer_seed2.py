"""W5 Amendment 2: BRIGHT-50 replication arm. `nnUNetTrainerW5Bright50Seed2` is `nnUNetTrainerW5Bright50` with ONE difference: the fixed seed (20261028 instead of 20261020). Nothing else is overridden (the AST test in tests/test_w5_seed2.py enforces it).
Installed by copying this file to `<venv_nnunet>/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerW5Seed2.py`; the original `nnUNetTrainerW5.py` is untouched (its hash is unchanged)."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5 import nnUNetTrainerW5Bright50

SEED_B50_S2 = 20261028      # fixed in PREREG Amendment 2, before any run of this arm


class nnUNetTrainerW5Bright50Seed2(nnUNetTrainerW5Bright50):
    SEED = SEED_B50_S2
