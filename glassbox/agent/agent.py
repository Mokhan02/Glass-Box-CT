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
    "Write a short research report from the tool results below. Rules: (1) every number must be copied from a tool result and "
    "followed by its source in brackets, e.g. [body_composition]; (2) never compute, estimate or round numbers yourself; "
    "(3) if the quality audit flags a structure, say it is flagged as likely wrong, quote its probability and name its drivers; if it could not assess a structure, say so; "
    "(4) if a tool failed or returned nothing, say so instead of guessing; (5) call these research measurements, never diagnoses "
    "or recommendations; (6) if measure_lesions returns tumour_liver_contrast, report it with its unit and basis, and if it returns contrast_caution, include that caution word for word "
    "together with its evidence_status, never turning it into a statement about this patient; (7) end with exactly: " + DISCLAIMER
)


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
    if not report.endswith(DISCLAIMER):
        report += "\n\n" + DISCLAIMER
    fixture = any(isinstance(e.get("result"), dict) and e["result"].get("fixture") for e in toolbox.evidence)
    if fixture:
        report = "**TEST FIXTURE DATA: made-up values, not real results.**\n\n" + report
    return {"scan_id": scan_id, "report": report, "evidence": toolbox.evidence, "fixture": fixture,
            **check_report(report, toolbox.evidence),
            "provider_host": cfg.provider_host, "models": {"fast": cfg.fast_model, "reasoning": cfg.model}}
