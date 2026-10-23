import numpy as np

from glassbox.seg import w3s2 as T


def _case():
    L = np.zeros((40, 40, 40), np.int16); L[5:35, 5:35, 5:35] = 1; L[15:25, 15:25, 15:25] = 2; L[:, :, 36:] = 3          # liver block, tumour cube, unannotated top slab
    pred = np.zeros(L.shape, bool); pred[15:20, 15:25, 15:25] = True; hu = np.full(L.shape, 60.0); hu[15:25, 15:25, 15:25] = 110.0
    return L, pred, hu


def test_token_groups_coverage_and_interior_distance_anisotropic_cells():
    L, pred, hu = _case(); sp = (1.0, 1.0, 2.0); st = T.GridStats(L, pred, hu, sp)
    xyz = np.array([[20, 20, 20], [8, 8, 8], [20, 20, 38], [15, 20, 20]]); t = st.tokens(xyz, (8, 8, 4))
    assert t["is_tum"][0] and not t["is_tum"][1]                                                       # a tumour-centred cell is a TUMOUR token, a liver cell is not
    assert t["is_liv"][1] and not t["is_tum"][1]                                                      # liver-only cell
    assert not t["is_tum"][2] and not t["is_liv"][2]                                                  # unannotated slab: dropped
    assert t["pred_cov"][0] > 0 and t["pred_cov"][1] == 0                                              # half of the tumour cube is predicted
    assert t["edt"][0] > 2.0 and t["edt"][3] <= 6.0                                                    # centre of the cube is interior; the face-centred token is not deep
    assert abs(t["hu14"][0, 0] - 110.0) < 1e-3 and abs(t["hu14"][1, 0] - 60.0) < 1e-3                  # cell mean HU


def test_roi_is_liver_plus_tumour_dilated_10mm_in_mm():
    far = np.zeros((60, 10, 10), np.int16); far[0:5] = 1; rr = T.roi_mask(far, (1.0, 1.0, 1.0)); assert rr[14, 5, 5] and not rr[16, 5, 5]


def test_window_origins_flush_and_overlap():
    o = T.window_origins((100, 40, 96), (96, 96, 96), 0.5); assert (0, 0, 0) in o and any(a == 4 for a, _, _ in o)       # 100 > 96: starts 0 and 4 (flush), the small axes use one window
    g = T.window_origins((300,) * 3, (128,) * 3, 0.3); assert max(a for a, _, _ in g) == 300 - 128 and len({a for a, _, _ in g}) >= 3


def test_hu14_subsampling_is_exact_for_8_voxel_cells_and_close_for_16():
    rng = np.random.default_rng(0); L = np.zeros((70, 70, 70), np.int16); L[5:65, 5:65, 5:65] = 1; hu = rng.normal(60, 20, L.shape).astype(np.float32); st = T.GridStats(L, np.zeros(L.shape, bool), hu, (1.0, 1.0, 1.0)); xyz = np.array([[30, 30, 30]])
    f8 = st.hu14(xyz, np.array([8, 8, 8])); lo, hi = 26, 34; exact = np.percentile(hu[lo:hi, lo:hi, lo:hi], [5, 25, 50, 75, 95]); assert np.allclose(f8[0, 2:7], exact, atol=1e-4)
    f16 = st.hu14(xyz, np.array([16, 16, 16])); ex16 = np.percentile(hu[22:38, 22:38, 22:38], [5, 25, 50, 75, 95]); assert np.abs(f16[0, 2:7] - ex16).max() < 4.0          # 2x subsample of a 16^3 cell: sampling noise only
