from glassbox.pipeline.select import select_demos


def _r(i, cov, con, flag, l3=False, n=1):
    return {"id": i, "coverage": cov, "contrast": con, "flag": flag, "assessed": flag != "not assessable", "l3_available": l3, "n_lesions": n}


def test_each_slot_follows_its_rule_and_no_scan_is_used_twice():
    rows = [_r("a", 0.95, -20, "no flag"), _r("b", 0.05, 30, "likely wrong"), _r("c", 0.9, 25, "no flag"), _r("d", 0.9, -5, "likely wrong"), _r("e", 0.5, -5, "no flag", l3=True, n=0), _r("f", 0.99, -9, "no flag", l3=True)]
    s = {x["slot"]: x["row"] for x in select_demos(rows)}
    assert s["well_found_dark"]["id"] in ("a", "f") and s["bright_missed_flagged"]["id"] == "b" and s["bright_found"]["id"] == "c" and s["flag_false_alarm"]["id"] == "d" and s["body_composition"]["l3_available"]
    ids = [x["row"]["id"] for x in select_demos(rows) if x["row"]]; assert len(ids) == len(set(ids))


def test_a_missing_slot_is_reported_as_none_not_silently_filled():
    s = select_demos([_r("a", 0.95, -20, "no flag")]); assert s[1]["row"] is None and s[3]["row"] is None
