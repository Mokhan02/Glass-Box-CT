"""W2-M solver and reproduction check: re-run scripts/voxel_probe_analyze.py (unchanged) on the stored HCC token tables, optionally with a tighter LogisticRegression solver (tol, max_iter patched at construction).
Usage: python scripts/w2m_hcc_repro.py --mode default|tight --cohort all|single_acquisition|uid_route [--random-groups-from-trained] --out DIR [--cache /lambda/nfs/CTGate/glassbox/cache]"""
import argparse
import runpy
import sys

ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="default", choices=["default", "tight"]); ap.add_argument("--cohort", default="all"); ap.add_argument("--random-groups-from-trained", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--cache", default="/lambda/nfs/CTGate/glassbox/cache"); a = ap.parse_args()
if a.mode == "tight":
    import sklearn.linear_model as lm
    _fit = lm.LogisticRegression.fit

    def _tight_fit(self, X, y, sample_weight=None):      # patch fit (not __init__): sklearn introspects the __init__ signature
        self.tol = 1e-8; self.max_iter = 20000
        return _fit(self, X, y, sample_weight)
    lm.LogisticRegression.fit = _tight_fit
c = a.cache
argv = ["voxel_probe_analyze.py", "--vp", f"{c}/vp", "--prob", f"{c}/vp_prob", "--root", f"{c}/heldout_nii/hcc_tace_seg", "--acq", f"{c}/hcc_acq_check.csv", "--out", a.out, "--skip-recovery", "--cohort", a.cohort]
if a.random_groups_from_trained:
    argv.append("--random-groups-from-trained")
sys.argv = argv; runpy.run_path("scripts/voxel_probe_analyze.py", run_name="__main__")
