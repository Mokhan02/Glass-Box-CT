import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ts_runner_mask_extraction_and_failure_on_missing_class():
    import pytest
    spec = importlib.util.spec_from_file_location("tsr", os.path.join(HERE, "scripts", "w6_ts_run.py")); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    arr = np.zeros((6, 6, 6), np.int16); arr[1:3, 1:3, 1:3] = 1; arr[4:6, 4:6, 4:6] = 2; cm = {1: "liver_vessels", 2: "liver_tumor"}
    m = T.mask_from_multilabel(arr, cm, "liver_tumor"); assert m.sum() == 8 and m.dtype == bool and m[4, 4, 4] and not m[1, 1, 1]
    with pytest.raises(KeyError):
        T.mask_from_multilabel(arr, {1: "liver_vessels"}, "liver_tumor")      # a class map without the class is an error, never an empty mask
