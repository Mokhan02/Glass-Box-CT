import numpy as np

from glassbox.bodycomp.counterfactual import HU_MAX, HU_MIN, arm_targets, edit_lesion, feather_weights


def _vol():
    ct = np.full((20, 20, 20), 100.0, np.float32); m = np.zeros(ct.shape, bool); m[6:14, 6:14, 6:14] = True; ct[m] = 60.0
    return ct, m


def test_edit_never_touches_voxels_outside_the_mask():
    ct, m = _vol(); out, _ = edit_lesion(ct, m, 140.0)
    assert np.array_equal(out[~m], ct[~m]) and not np.array_equal(out[m], ct[m])
    assert not np.shares_memory(out, ct) and ct[m].mean() == 60.0  # original untouched


def test_feather_is_one_voxel_half_weight_on_the_border_and_full_inside():
    _, m = _vol(); w = feather_weights(m)
    assert w[~m].max() == 0.0 and set(np.unique(w[m])) == {0.5, 1.0}
    assert w[6, 10, 10] == 0.5 and w[10, 10, 10] == 1.0 and (w[m] == 1.0).sum() == 6 ** 3


def test_offset_is_constant_so_realised_mean_lands_between_start_and_target_and_is_recorded():
    ct, m = _vol(); out, info = edit_lesion(ct, m, 140.0)
    assert info["T"] == 60.0 and info["offset"] == 80.0
    assert 60.0 < info["realised_mean"] < 140.0 and info["flagged"]  # border at half weight leaves the mean short of the target by more than 5 HU
    assert np.isclose(out[10, 10, 10], 140.0) and np.isclose(out[6, 10, 10], 100.0)


def test_values_are_clipped_to_the_valid_hu_range():
    ct, m = _vol(); out, _ = edit_lesion(ct, m, 5000.0); assert out[m].max() <= HU_MAX
    out2, _ = edit_lesion(ct, m, -5000.0); assert out2[m].min() >= HU_MIN


def test_arm_targets_follow_the_preregistered_definitions():
    arm, edit, ctrl = arm_targets(T=80.0, L=110.0)           # dark tumour, c = -30
    assert arm == "A" and edit == 140.0 and ctrl == 20.0     # flip: L + |c| ; control: T - 2|c|
    arm, edit, ctrl = arm_targets(T=140.0, L=110.0)          # bright tumour, c = +30
    assert arm == "B" and edit == 80.0 and ctrl == 200.0     # flip: L - |c| ; control: T + 2|c|
    assert arm_targets(100.0, 100.0) is None
