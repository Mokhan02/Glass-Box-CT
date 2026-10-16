import numpy as np

from glassbox.audit.contrast import case_lesions


def _kidney_phantom(lesion_hu=110.0, kidney_hu=150.0, shape=(140, 100, 60), sp=(1.0, 1.0, 1.0)):
    """Ipsilateral kidney block with a 9 mm-radius lesion at its centre, a far (contralateral) bright kidney, a second Mass and a bright renal vessel."""
    ct = np.full(shape, -100.0); kid = np.zeros(shape, bool)
    kid[10:60, 20:80, 10:50] = True                      # ipsilateral kidney
    kid[95:135, 20:80, 10:50] = True                     # contralateral kidney, > 20 mm away from the lesion
    ct[kid] = kidney_hu; ct[95:135, 20:80, 10:50] = 400.0
    g = np.indices(shape).astype(float)
    lesion = ((g[0] - 35) ** 2 + (g[1] - 50) ** 2 + (g[2] - 30) ** 2) <= 81
    second = ((g[0] - 52) ** 2 + (g[1] - 50) ** 2 + (g[2] - 30) ** 2) <= 16          # a second Mass inside the shell
    vessel = (np.abs(g[0] - 35) <= 2) & (np.abs(g[1] - 70) <= 1) & (np.abs(g[2] - 30) <= 12)   # a bright renal vessel inside the shell
    ct[lesion] = lesion_hu; ct[second] = 20.0; ct[vessel] = 500.0
    return ct, lesion, second, vessel, kid, sp


def test_kidney_bright_and_dark_lesions_with_other_masses_vessels_and_far_kidney():
    for lesion_hu, want in ((190.0, 40.0), (110.0, -40.0)):
        ct, les, second, vessel, kid, sp = _kidney_phantom(lesion_hu)
        rows = case_lesions(ct, les | second, kid, sp, n_lesions=None, exclude=vessel)
        row = rows[0]
        assert row["rank"] == 1 and abs(row["liver_hu"] - 150.0) < 1e-6 and abs(row["contrast_hu"] - want) < 1e-6 and not row["fallback"]


def test_without_exclusions_the_reference_is_contaminated():
    ct, les, second, vessel, kid, sp = _kidney_phantom(190.0)
    row = case_lesions(ct, les, kid, sp)[0]                 # second Mass and vessel left in
    assert abs(row["liver_hu"] - 150.0) > 0.05
