"""Tool-calling agent: a fast model gathers evidence, a reasoning model writes the report, and the report is
checked against the evidence before it is returned. Provider-agnostic: the client and model IDs come from config."""
from __future__ import annotations

import json

from glassbox.agent.config import LLMConfig
from glassbox.agent.grounding import check_report
from glassbox.agent.tools import TOOL_SCHEMAS, ToolBox

DISCLAIMER = "Research use only, not for clinical decisions."

GATHER_PROMPT = (
    "You are a CT research-measurement assistant. For the given scan, call the tools to collect: the segmentation, "
    "body composition, lesion measurements and the quality audit. Do not write the report yet and do not invent numbers."
)
REPORT_PROMPT = (
    "Write a short research summary of this CT from the tool results below, for a reader who also sees the numbers in a panel next to it. "
    "At most about 150 words of plain prose, no tables, do NOT list organ volumes or every lesion. Cover, in this order: "
    "(a) whether the tumour segmentation can be trusted according to the quality audit (say if it is flagged as likely wrong, quote its probability, and name its drivers; "
    "if it could not assess, say so); (b) the largest tumour and the sum of the target diameters, with units; (c) if measure_lesions returns tumour_liver_contrast, one sentence on what that contrast is; "
    "(d) one sentence on muscle at L3 only if body_composition is available. "
    "Rules: (1) every number must be copied from a tool result and followed by its source in brackets, e.g. [measure_lesions]; "
    "(2) never compute, estimate or round numbers yourself, and write negative numbers with the ASCII minus sign '-'; "
    "(3) if a tool failed or returned nothing, say so instead of guessing; (4) call these research measurements, never diagnoses or recommendations; "
    "(5) if measure_lesions returns tumour_liver_contrast, report it with its unit and basis, and if it returns contrast_caution, include that caution word for word "
    "together with its evidence_status, never turning it into a statement about this patient; "
    "(6) if measure_lesions returns standing_limitation, include it word for word in every report, including when no lesion was found; "
    "(7) end with exactly: " + DISCLAIMER
)


def _verbatim_notes(report: str, evidence: list[dict]) -> str:
    """The standing limitation and the contrast caution must be in every report that has them in its tool results; a model that leaves them out gets them appended."""
    add = []
    for e in evidence:
        r = e.get("result") if e.get("ok") else None
        if isinstance(r, dict) and r.get("standing_limitation"):
            notes = [r["standing_limitation"], r.get("contrast_caution"), (r.get("contrast_context") or {}).get("evidence_status") if r.get("contrast_caution") else None]
            add += [n for n in notes if n and n not in report]
    return report.rstrip() + ("\n\n" + " ".join(add) if add else "")


def run_agent(scan_id: str, client, cfg: LLMConfig, toolbox: ToolBox, max_steps: int = 8) -> dict:
    messages = [{"role": "system", "content": GATHER_PROMPT}, {"role": "user", "content": f"Scan id: {scan_id}"}]
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=cfg.fast_model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            rec = toolbox.call(c.function.name, c.function.arguments)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(rec)})
    final = client.chat.completions.create(model=cfg.model, messages=[
        {"role": "system", "content": REPORT_PROMPT},
        {"role": "user", "content": f"Scan id: {scan_id}\nTool results:\n{json.dumps(toolbox.evidence, indent=1)}"}])
    report = (final.choices[0].message.content or "").strip()
    report = _verbatim_notes(report, toolbox.evidence)
    if not report.endswith(DISCLAIMER):
        report += "\n\n" + DISCLAIMER
    fixture = any(isinstance(e.get("result"), dict) and e["result"].get("fixture") for e in toolbox.evidence)
    if fixture:
        report = "**TEST FIXTURE DATA: made-up values, not real results.**\n\n" + report
    return {"scan_id": scan_id, "report": report, "evidence": toolbox.evidence, "fixture": fixture,
            **check_report(report, toolbox.evidence),
            "provider_host": cfg.provider_host, "models": {"fast": cfg.fast_model, "reasoning": cfg.model}}
