"""W6b distillation loss tests (results/w6b_distill/PREREG.md section 3): (a) student == teacher gives zero gradient, (b) an empty-teacher class with a faint student haze gives a small loss (not about 1), (c) the loss grows smoothly with disagreement; plus the class sampler."""
import math

import pytest
import torch

from glassbox.seg import w6b_loss as L


def logit(p):
    return math.log(p / (1 - p))


def test_a_student_equals_teacher_zero_gradient_and_zero_kl():
    torch.manual_seed(0); t = torch.randn(4, 1, 8, 8, 8) * 3; s = t.clone().requires_grad_(True)
    bce, kl = L.distill_bce(s, t); bce.backward()
    assert float(s.grad.abs().max()) < 1e-9 and abs(float(kl)) < 1e-7      # gradient exactly zero; the value is only the teacher's entropy, the KL is zero


def test_b_empty_teacher_class_with_faint_haze_is_a_small_loss():
    t = torch.full((1, 1, 16, 16, 16), -20.0)      # teacher probability about 2e-9: the class is empty
    s = torch.full((1, 1, 16, 16, 16), logit(0.01), requires_grad=True)      # student haze of 0.01 everywhere
    bce, kl = L.distill_bce(s, t); assert 0.0 < float(bce) < 0.02 and float(kl) < 0.02      # about -log(0.99) = 0.010, nowhere near 1


def test_c_loss_grows_smoothly_with_disagreement():
    torch.manual_seed(1); t = torch.randn(2, 1, 8, 8, 8) * 2; vals = []
    for d in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        vals.append(float(L.distill_bce(t + d, t)[1]))
        vals.append(float(L.distill_bce(t - d, t)[1]))
    up = [vals[i] for i in range(0, len(vals), 2)]; dn = [vals[i] for i in range(1, len(vals), 2)]
    for seq in (up, dn):
        assert seq[0] < 1e-7 and all(b > a for a, b in zip(seq, seq[1:]))      # strictly increasing from zero, in both directions
    ratios = [b / a for a, b in zip(up[1:], up[2:])]; assert all(r < 8 for r in ratios)      # no jump: roughly quadratic growth for small gaps


def test_teacher_receives_no_gradient_and_is_not_changed():
    t = torch.randn(1, 1, 4, 4, 4, requires_grad=True); s = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
    L.distill_bce(s, t)[0].backward(); assert t.grad is None and s.grad is not None


def test_sampler_distinct_from_the_set_and_reproducible():
    classes = [c for c in range(1, 133) if c != 26]
    g1 = torch.Generator().manual_seed(7); g2 = torch.Generator().manual_seed(7)
    a = L.sample_distill_classes(g1, classes, 16); b = L.sample_distill_classes(g2, classes, 16)
    assert a == b and len(set(a)) == 16 and set(a) <= set(classes) and 0 not in a and 26 not in a
    seen = set()
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        seen |= set(L.sample_distill_classes(g, classes, 16))
    assert seen == set(classes)      # every class is reachable
    with pytest.raises(AssertionError):
        L.sample_distill_classes(g, classes, 0)
