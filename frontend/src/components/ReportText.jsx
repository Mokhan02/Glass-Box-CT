import { Fragment } from "react";

const UNIT = "(?:cm2|cm3|cm²|cm³|mm|HU|%)";
const TOOLS = "(?:segment_ct|body_composition|measure_lesions|quality_audit)";

/** The written (grounded) report. Numbers with a unit are clickable and jump the viewer to where they were measured. */
export default function ReportText({ text, summary, onJump }) {
  const target = (tool, valueText) => {
    if (!summary) return null; const v = parseFloat(valueText);
    if (tool === "body_composition" && summary.body_composition && summary.body_composition.available) return summary.body_composition.l3_voxel_in_file;
    const ls = summary.lesions && summary.lesions.lesions;
    if (ls && ls.length && (tool === "measure_lesions" || tool === "quality_audit")) { const hit = ls.find((l) => Math.abs(l.longest_axial_diameter.value - v) < 0.06 || Math.abs(l.volume.value - v) < 0.06); return (hit || ls[0]).voxel_in_file; }
    return null;
  };
  const para = (p, i) => {
    const first = (p.match(new RegExp("\\[(" + TOOLS + ")\\]")) || [])[1], parts = [], re = new RegExp("(\\*\\*.+?\\*\\*)|(-?\\d+(?:\\.\\d+)?\\s?" + UNIT + ")|(\\[" + TOOLS + "\\])", "g"); let last = 0, m;
    while ((m = re.exec(p))) {
      if (m.index > last) parts.push(p.slice(last, m.index));
      if (m[1]) parts.push(<strong key={m.index} className="font-medium text-ink">{m[1].slice(2, -2)}</strong>);
      else if (m[2]) { const t = target(first, m[2]); parts.push(t ? <button key={m.index} className="num focus-ring rounded px-0.5 text-accent underline decoration-dotted underline-offset-2 hover:bg-accent/10" onClick={() => onJump(t)} title="Show where this was measured">{m[2]}</button> : <span key={m.index} className="num">{m[2]}</span>); }
      else parts.push(<span key={m.index} className="num text-[11px] text-ink-dim">{m[3]}</span>);
      last = m.index + m[0].length;
    }
    if (last < p.length) parts.push(p.slice(last));
    return <p key={i} className="mb-2.5">{parts.map((x, k) => <Fragment key={k}>{x}</Fragment>)}</p>;
  };
  return <div className="text-[13px] leading-relaxed text-ink-dim">{text.split(/\n\s*\n/).map(para)}</div>;
}
