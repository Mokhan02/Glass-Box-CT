import { Fragment } from "react";

const UNIT = "(?:cm2|cm3|cm²|cm³|mm|HU|%)";
const TOOLS = "(?:segment_ct|body_composition|measure_lesions|quality_audit)";
// Paragraphs the page already shows elsewhere (key numbers, verdict, "Known limits") are not repeated in the prose.
const SHOWN_ELSEWHERE = ["**Limitation**", "**Known bias of this measurement**", "**Caution (research finding)", "**Tumour-minus-liver contrast**", "**Quality audit, hepatic tumour**"];

export function stripShownElsewhere(text, summary) {
  const L = (summary && summary.lesions) || {}; let t = text;
  [L.standing_limitation, L.contrast_caution, L.contrast_context && L.contrast_context.evidence_status].filter(Boolean).forEach((s) => { t = t.split(s).join(""); });  // verbatim sentences the agent was told to include
  return t.split(/\n\s*\n/).map((p) => p.trim()).filter((p) => p && !SHOWN_ELSEWHERE.some((k) => p.startsWith(k)) && !/^\[(measure_lesions|quality_audit)\]\s*$/.test(p)).join("\n\n");
}

/** The written report. Numbers with a unit are clickable (jump the viewer to where they were measured); the tool each statement came from shows on hover. */
export default function ReportText({ text, summary, onJump }) {
  const target = (tool, valueText) => {
    if (!summary) return null; const v = parseFloat(valueText);
    if (tool === "body_composition" && summary.body_composition && summary.body_composition.available) return summary.body_composition.l3_voxel_in_file;
    const ls = summary.lesions && summary.lesions.lesions;
    if (ls && ls.length && (tool === "measure_lesions" || tool === "quality_audit")) { const hit = ls.find((l) => Math.abs(l.longest_axial_diameter.value - v) < 0.06 || Math.abs(l.volume.value - v) < 0.06); return (hit || ls[0]).voxel_in_file; }
    return null;
  };
  const para = (p, i) => {
    const first = (p.match(new RegExp("\\[(" + TOOLS + ")\\]")) || [])[1], prov = first ? `Source: ${first}` : undefined, parts = [];
    const re = new RegExp("(\\*\\*.+?\\*\\*)|(-?\\d+(?:\\.\\d+)?\\s?" + UNIT + ")|(\\s?\\[" + TOOLS + "\\])", "g"); let last = 0, m;
    while ((m = re.exec(p))) {
      if (m.index > last) parts.push(p.slice(last, m.index));
      if (m[1]) parts.push(<strong key={m.index} className="font-medium text-ink">{m[1].slice(2, -2)}</strong>);
      else if (m[2]) { const t = target(first, m[2]); parts.push(t ? <button key={m.index} title={prov} className="num focus-ring rounded px-0.5 text-accent underline decoration-dotted underline-offset-2 hover:bg-accent/10" onClick={() => onJump(t)}>{m[2]}</button> : <span key={m.index} title={prov} className="num text-ink">{m[2]}</span>); }
      last = m.index + m[0].length;  // [tool] tags are dropped from the text; provenance is on hover
    }
    if (last < p.length) parts.push(p.slice(last));
    return <p key={i} title={prov} className="mb-2.5">{parts.map((x, k) => <Fragment key={k}>{x}</Fragment>)}</p>;
  };
  const clean = text.replace(/\u2212/g, "-").replace(/(?<![\d.])\u2013(?=\d)/g, "-");  // a typographic minus or en dash before a number is a minus sign: keep it inside the clickable number
  return <div className="text-[13px] leading-relaxed text-ink-dim">{clean.split(/\n\s*\n/).map(para)}</div>;
}
