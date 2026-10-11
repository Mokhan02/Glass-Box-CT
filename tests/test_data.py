from glassbox.data.registry import is_in_distribution
from glassbox.data.splits import split_patients


def test_split_is_deterministic_and_disjoint():
    ids = [f"p{i}" for i in range(100)]
    a = split_patients(ids)
    assert a == split_patients(reversed(ids))
    assert set(a) == set(ids)
    counts = {s: list(a.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_duplicate_scans_of_one_patient_share_a_split():
    a = split_patients(["p1", "p1", "p2", "p3"])
    assert len(a) == 3


def test_in_distribution_flags():
    assert is_in_distribution("totalsegmentator")
    assert not is_in_distribution("c4kc_kits")
    assert is_in_distribution("never_heard_of_it")


def test_results_table_flag_is_derived_not_supplied(tmp_path):
    import csv

    from glassbox.data.results import ResultsTable

    t = ResultsTable(["auroc"])
    t.add("totalsegmentator", 40, auroc=0.9)
    t.add("c4kc_kits", 20, auroc=0.8)
    try:
        t.add("c4kc_kits", 5, in_distribution=False)
        raise AssertionError("flag must not be settable")
    except KeyError:
        pass
    assert [r["dataset"] for r in t.held_out()] == ["c4kc_kits"]
    t.to_csv(tmp_path / "r.csv")
    rows = list(csv.DictReader(open(tmp_path / "r.csv")))
    assert rows[0]["in_distribution"] == "True" and rows[1]["in_distribution"] == "False"


def test_every_held_out_name_used_by_scripts_is_registered_as_held_out():
    from glassbox.data.registry import DATASETS

    for name in ("c4kc_kits", "hcc_tace_seg"):
        assert name in DATASETS and not DATASETS[name].in_distribution
