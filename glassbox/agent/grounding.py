"""Check a report against the tool evidence. Two failure kinds, both reported:
  unsupported_numbers  a measurement in the report that matches no evidence value
  missing_units        a number whose evidence value carries a unit (HU, mm, cm2, ...) but is printed without it
Evidence quantities are {"value": v, "unit": u} pairs (see tools.qty). Dimensionless numbers (indices, probabilities, fractions) are plain.
Checked in the report: numbers with a decimal point and numbers followed by a unit. Plain integers are ignored.
A number is supported if it equals an evidence number after rounding to 0, 1 or 2 decimals; a '%' figure may be the evidence value times 100."""
from __future__ import annotations

import re

UNIT = r"(?:cm\s*²|cm2|cm\^2|cm\s*³|cm3|cm\^3|mm|HU|%|Hounsfield\s+units?)"
NUM = re.compile(rf"(?<![\w.])(-?\d+(?:\.\d+)?)(\s*{UNIT})?", re.IGNORECASE)


def canon_unit(u: str) -> str:
    u = re.sub(r"\s+", "", u.lower())
    return {"cm²": "cm2", "cm^2": "cm2", "cm³": "cm3", "cm^3": "cm3", "hounsfieldunit": "hu", "hounsfieldunits": "hu"}.get(u, u)


def is_qty(o) -> bool:
    return isinstance(o, dict) and set(o) >= {"value", "unit"} and isinstance(o["value"], (int, float)) and not isinstance(o["value"], bool)


def _walk(obj, nums: set, qtys: list) -> None:
    if is_qty(obj):
        nums.add(float(obj["value"]))
        if obj["unit"]:
            qtys.append((float(obj["value"]), canon_unit(obj["unit"])))
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        nums.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, nums, qtys)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, nums, qtys)
    elif isinstance(obj, str):
        for m in re.finditer(r"-?\d+(?:\.\d+)?", obj):
            nums.add(float(m.group()))


def _same(e: float, c: float) -> bool:
    return abs(e - c) < 1e-9 or any(abs(round(e, d) - c) < 1e-9 for d in (0, 1, 2))


def _collect(evidence):
    nums, qtys = set(), []
    _walk([e.get("result") for e in evidence if e.get("ok", True)], nums, qtys)
    _walk([e.get("args") for e in evidence], nums, [])  # scan and feature ids may be quoted
    return nums, qtys


NAMES = re.compile(r"RECIST\s*1\.1", re.IGNORECASE)  # a criteria name, not a measurement


def check_report(report: str, evidence: list[dict]) -> dict:
    report = NAMES.sub("RECIST", report).replace("\u2212", "-").replace("\u2011", "-")  # typographic minus / non-breaking hyphen are negative signs, not part of a different number
    report = re.sub(r"(?<![\d.])\u2013(?=\d)", "-", report)  # an en dash directly before a number (not inside a range like 5\u201310) is a minus sign
    nums, qtys = _collect(evidence)
    unsupported, missing = [], []
    for m in NUM.finditer(report):
        text, unit = m.group(1), canon_unit(m.group(2) or "")
        if "." not in text and not unit:
            continue
        x = float(text)
        cands = {x} | ({x / 100} if unit == "%" else set())
        if not any(_same(e, c) for e in nums for c in cands):
            unsupported.append((text + (" " + unit if unit else "")).strip())
            continue
        need = {u for v, u in qtys if _same(v, x)}
        if need and unit not in need and not (unit == "%" and not need):
            missing.append(f"{text} (needs {'/'.join(sorted(need))})" + (f", printed with {unit}" if unit else ", printed bare"))
    return {"unsupported_numbers": unsupported, "missing_units": missing, "ok": not unsupported and not missing}


def unsupported_numbers(report: str, evidence: list[dict]) -> list[str]:  # kept for callers of the earlier API
    return check_report(report, evidence)["unsupported_numbers"]
