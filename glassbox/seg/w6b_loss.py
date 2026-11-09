"""W6b self-distillation pieces that need no bundle (results/w6b_distill/PREREG.md section 3): the voxelwise BCE against the teacher's soft probabilities, and the class sampler.
Loss: mean over the K sampled classes and all voxels of BCE(student logit, teacher probability), with teacher probability = sigmoid(teacher logit). At student == teacher the GRADIENT is exactly zero (d/dz = p_s - p_t) while the VALUE is the teacher's mean binary entropy; the entropy-free value (the KL divergence, same gradient) is returned too for monitoring."""
import torch
import torch.nn.functional as F


def distill_bce(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    """Returns (bce, kl): both scalars, means over every element (classes x voxels). The teacher is treated as a constant."""
    t = teacher_logits.detach().float(); s = student_logits.float(); pt = torch.sigmoid(t)
    bce = F.binary_cross_entropy_with_logits(s, pt, reduction="mean"); ent = F.binary_cross_entropy_with_logits(t, pt, reduction="mean")
    return bce, bce - ent


def sample_distill_classes(gen: torch.Generator, classes, k: int):
    """k distinct classes drawn uniformly without replacement from `classes` with the given CPU generator (sorted list of ints)."""
    classes = list(classes); assert 0 < k <= len(classes) and len(set(classes)) == len(classes)
    idx = torch.randperm(len(classes), generator=gen)[:k].tolist(); return sorted(classes[i] for i in idx)
