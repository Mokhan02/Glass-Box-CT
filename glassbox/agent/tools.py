"""The five tools the agent may call, as OpenAI function-calling schemas, plus the dispatcher.
Tools return plain JSON. The agent never computes a number itself: every figure in a report must come from one of these results."""
from __future__ import annotations

import json
from typing import Any, Protocol

from glassbox.seg.prompts import require_tumour_prompts

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "segment_ct", "description": "Segment a CT scan with NV-Segment-CT. Returns the structures found and the mask file. Tumour classes are prompted explicitly because the model's segment-everything mode omits them.",
        "parameters": {"type": "object", "properties": {"scan_id": {"type": "string"}}, "required": ["scan_id"]}}},
    {"type": "function", "function": {
        "name": "body_composition", "description": "Psoas and paraspinal muscle area (cm2) and mean density (HU) at the L3 slice. Research measurement only.",
        "parameters": {"type": "object", "properties": {"scan_id": {"type": "string"}}, "required": ["scan_id"]}}},
    {"type": "function", "function": {
        "name": "measure_lesions", "description": "Lesions with longest axial diameter (mm), organ, and a RECIST target or non-target suggestion. Always prompts the hepatic tumour class (26) and other tumour classes explicitly, and reports which classes it prompted.",
        "parameters": {"type": "object", "properties": {"scan_id": {"type": "string"}}, "required": ["scan_id"]}}},
    {"type": "function", "function": {
        "name": "quality_audit", "description": "Probability that the hepatic-tumour segmentation is wrong, a quality flag, and the factors that drove it (predicted size and model confidence). Reports plainly when a structure cannot be assessed.",
        "parameters": {"type": "object", "properties": {"scan_id": {"type": "string"}}, "required": ["scan_id"]}}},
]


def qty(value: float, unit: str) -> dict:
    """A measurement with its unit. Every physical quantity a tool returns must be one of these."""
    return {"value": value, "unit": unit}


DIMENSIONLESS_KEY_HINTS = ("probability", "fraction", "confidence", "score")  # may be bare floats


def bare_floats(obj, key: str = "") -> list[str]:
    """Paths of float values that should have been qty() pairs. ints (indices, counts, ids) and dimensionless keys are allowed."""
    if isinstance(obj, dict):
        if set(obj) >= {"value", "unit"}:
            return []
        return [p for k, v in obj.items() for p in bare_floats(v, k)]
    if isinstance(obj, list):
        return [p for v in obj for p in bare_floats(v, key)]
    if isinstance(obj, float) and not any(h in key.lower() for h in DIMENSIONLESS_KEY_HINTS):
        return [f"{key}={obj}"]
    return []


class Backend(Protocol):
    def segment_ct(self, scan_id: str) -> dict: ...
    def body_composition(self, scan_id: str) -> dict: ...
    def measure_lesions(self, scan_id: str) -> dict: ...
    def quality_audit(self, scan_id: str) -> dict: ...


class ToolBox:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.evidence: list[dict] = []  # every call and result, in order: the report's only permitted source of numbers

    def call(self, name: str, arguments: str | dict) -> dict:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if name not in {t["function"]["name"] for t in TOOL_SCHEMAS}:
                raise ValueError(f"unknown tool {name!r}")
            result = getattr(self.backend, name)(**args)
            if name == "measure_lesions":  # an empty lesion list only means 'no tumour' if the tumour class was actually asked for
                require_tumour_prompts(result.get("prompted_classes"))
            bad = bare_floats(result)
            if bad:  # a tool must say what unit a measurement is in, so the report can be checked for it
                raise ValueError(f"result has measurements without units: {bad[:3]}")
            out = {"ok": True, "result": result}
        except Exception as e:  # a failing tool must not crash the agent; the model sees the error and says so
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        record = {"tool": name, "args": args if "args" in locals() else {}, **out}
        self.evidence.append(record)
        return record
