"""Choosing the five demo scans by stated rules, including cases where the quality flag is wrong or unneeded, so the demo does not overstate the flag.
Input rows (one per scan): id, coverage (fraction of the reference tumour the model found), contrast (HU, tumour minus local liver), flag ('likely wrong' | 'no flag' | 'not assessable'),
assessed (bool), l3_available (bool), n_lesions, shift_matters (bool: the largest reference lesion is covered < 10% at the default threshold and >= 50% at the cross-validated shifted threshold; exploratory).
Scans with the L3 level in view are preferred in every slot (tie-break only; no rule is relaxed). Pure function, unit-tested."""
from __future__ import annotations

SLOTS = [
    ("well_found_dark", "A tumour darker than the liver that the model found well, with no flag", "The typical good case: model found most of the reference tumour and the flag stayed quiet."),
    ("bright_missed_flagged", "A tumour brighter than the liver that the model largely missed, caught by the quality flag", "The case the project is built around: most of the reference tumour was not found, and the flag raised it."),
    ("shift_matters", "A tumour the model missed at its default setting that a recalibrated threshold recovers", "Where the threshold finding matters: at the default threshold the model found almost none of the reference tumour, and a single threshold shift (research finding, exploratory) finds most of it."),
    ("bright_found", "A tumour brighter than the liver that the model found well", "A counter-example: brighter tumours are missed more often, not always."),
    ("flag_false_alarm", "A case the flag marked as likely wrong although the model found most of the tumour", "The flag is not perfect: it also raises false alarms."),
]


def _first(rows, pred, key=lambda r: r["id"]):
    c = sorted((r for r in rows if pred(r)), key=key); return c[0] if c else None


def select_demos(rows: list[dict]) -> list[dict]:
    used, out = set(), []
    rules = {
        "well_found_dark": (lambda r: r["n_lesions"] and r["assessed"] and r["coverage"] >= 0.8 and r["contrast"] < 0 and r["flag"] == "no flag", lambda r: -r["coverage"]),
        "bright_missed_flagged": (lambda r: r["n_lesions"] and r["assessed"] and r["contrast"] > 0 and r["coverage"] < 0.3 and r["flag"] == "likely wrong", lambda r: abs(r["coverage"] - 0.10)),
        "shift_matters": (lambda r: r["n_lesions"] and r["assessed"] and r.get("shift_matters"), lambda r: r["coverage"]),
        "bright_found": (lambda r: r["n_lesions"] and r["assessed"] and r["contrast"] > 0 and r["coverage"] >= 0.8 and r["flag"] == "no flag", lambda r: -r["coverage"]),
        "flag_false_alarm": (lambda r: r["n_lesions"] and r["assessed"] and r["coverage"] >= 0.8 and r["flag"] == "likely wrong", lambda r: -r["coverage"]),
    }
    for slot, title, why in SLOTS:
        pred, key = rules[slot]; pick = _first([r for r in rows if r["id"] not in used], pred, lambda r, k=key: (not r["l3_available"], k(r), r["id"]))
        out.append({"slot": slot, "title": title, "why": why, "row": pick}); used.add(pick["id"]) if pick else None
    return out
