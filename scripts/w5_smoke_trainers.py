"""W5 5-epoch smoke trainers (no change to the committed W5 trainers): the same classes with num_epochs = 5, and the transform's core call logged to $W5_EDIT_LOG (one line per sample: edited?, tumour in patch?, deltas).
Installed next to nnUNetTrainerW5.py as `nnUNetTrainerW5Smoke.py`; results land in their own folders (trainer name differs), so nothing mixes with the real arms. The logging wrapper replaces `shift_tumour_contrast` in the
nnUNetTrainerW5 module before the dataloader workers fork; the transform class itself is the real one."""
import os

from nnunetv2.training.nnUNetTrainer import nnUNetTrainerW5 as W5
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5 import nnUNetTrainerW5Bright50, nnUNetTrainerW5Sham50

_orig = W5.shift_tumour_contrast


def _logged(image, seg, arm, rng, p):
    new, info = _orig(image, seg, arm, rng, p)
    path = os.environ.get("W5_EDIT_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write("%s %d %d %s\n" % (arm, int(bool((seg == W5.TUMOUR).any())), int(info["applied"]), ",".join("%.3f" % c["delta"] for c in info["components"] if "delta" in c)))
    return new, info


W5.shift_tumour_contrast = _logged


class _Smoke:
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.num_epochs = 5
        self.print_to_log_file("W5 SMOKE: 5 epochs; edit log $W5_EDIT_LOG")


class nnUNetTrainerW5SmokeBright50(_Smoke, nnUNetTrainerW5Bright50):
    pass


class nnUNetTrainerW5SmokeSham50(_Smoke, nnUNetTrainerW5Sham50):
    pass
