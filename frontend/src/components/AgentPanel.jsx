import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import Measure from "./Measure";
import Chip from "./Chip";
import ToolSteps from "./ToolSteps";
import RecalToggle from "./RecalToggle";
import ReportText from "./ReportText";
import { EASE } from "../lib/motion";

const fade = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.25, ease: EASE } };

function Section({ title, children, className = "" }) {
  return <section className={`border-t border-white/[0.07] py-5 ${className}`}><h3 className="eyebrow mb-3">{title}</h3>{children}</section>;
}

export default function AgentPanel({ meta, all, summary, report, error, mode, setMode, onJump }) {
  const [finished, setFinished] = useState(false);
  const les = (summary && summary.lesions && summary.lesions.lesions) || [];
  const q = summary && summary.quality_audit && summary.quality_audit.structures && summary.quality_audit.structures[0];
  const bc = summary && summary.body_composition, L = summary && summary.lesions;
  const targets = useMemo(() => les.filter((l) => l.recist === "target candidate"), [les]);
  const sum = targets.reduce((a, l) => a + l.longest_axial_diameter.value, 0), vol = les.reduce((a, l) => a + l.volume.value, 0);
  const bad = q && q.flag === "likely wrong";
  const drivers = q && q.drivers ? q.drivers.slice(0, 2).map((d) => `${d.name} (${d.pushes_toward})`).join("; ") : "";
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[17px] font-medium tracking-tight">Agent</h2>
        <span className="text-[12px] text-ink-dim">{report ? (report.mode === "agent" ? `language model: ${report.provider_host || "configured provider"}` : "template report · no language model called") : "working…"}</span>
      </div>
      <div className="mt-4"><ToolSteps evidence={report && report.evidence} done={!!report} onFinished={() => setFinished(true)} /></div>
      {error ? <p role="alert" className="mt-4 text-failure">{error}</p> : null}

      {report && finished ? (
        <motion.div {...fade} className="mt-5">
          <Section title="Measurements" className="border-t-0 pt-2">
            {les.length ? (
              <div className="grid grid-cols-2 gap-x-6 gap-y-5">
                <Measure label="Largest tumour" value={les[0].longest_axial_diameter.value} unit="mm" hint={`${les.length} predicted lesion${les.length > 1 ? "s" : ""}`} />
                <Measure label="Sum of diameters" value={sum} unit="mm" hint={`${targets.length} RECIST target${targets.length === 1 ? "" : "s"}`} />
                <Measure label="Predicted tumour volume" value={vol} unit="cm3" size="text-[26px]" />
                {L.tumour_liver_contrast ? <Measure label="Tumour minus liver" value={L.tumour_liver_contrast.value} unit="HU" size="text-[26px]" hint="on the predicted tumour" /> : null}
              </div>
            ) : <p className="text-[13px] leading-relaxed text-ink-dim">{(L && L.note) || "No tumour predicted."}</p>}
            {bc && bc.available ? <p className="mt-4 text-[12px] text-ink-dim">Muscle at L3 (psoas and paraspinal): <span className="num text-ink">{bc.psoas_paraspinal_area.value}</span> <span className="unit">cm2</span> · <span className="num text-ink">{bc.mean_muscle_density.value}</span> <span className="unit">HU</span></p> : null}
          </Section>

          {q ? (
            <Section title="Quality flag">
              <div className="flex flex-wrap items-center gap-3">
                {q.assessed ? <Chip kind={bad ? "flagged" : "good"}>{bad ? "Flagged: likely wrong" : "No flag"}</Chip> : <Chip kind="flagged">Cannot assess</Chip>}
                {q.assessed ? <span className="text-[13px] text-ink-dim">estimated probability the segmentation is wrong <span className="num text-ink">{q.failure_probability}</span></span> : null}
              </div>
              <p className="mt-2.5 text-[13px] leading-relaxed text-ink-dim">{q.assessed ? (drivers ? `Main drivers: ${drivers}. ` : "") + (q.caveat || "") : q.reason}</p>
            </Section>
          ) : null}

          <Section title="Threshold">
            <RecalToggle recal={summary.recalibrated} mode={mode} setMode={setMode} />
          </Section>

          <Section title="Read this first">
            <div className="rounded-xl border border-white/[0.12] bg-white/[0.02] p-4" style={{ borderLeft: "3px solid #F5B54A" }}>
              {L && L.contrast_caution ? <p className="mb-2.5 text-[13px] leading-relaxed text-ink">{L.contrast_caution}</p> : null}
              {L && L.standing_limitation ? <p className="text-[13px] leading-relaxed text-ink-dim">{L.standing_limitation}</p> : null}
              {L && L.contrast_caution ? <p className="mt-2.5 text-[12px] leading-relaxed text-ink-dim">{L.contrast_context && L.contrast_context.evidence_status}</p> : null}
            </div>
          </Section>

          <Section title="Written report">
            <details className="group">
              <summary className="focus-ring mb-3 w-fit cursor-pointer list-none rounded text-[12px] text-accent">Show the written report</summary>
              <ReportText text={report.report} summary={summary} onJump={onJump} />
              <p className="text-[12px] text-ink-dim">{report.mode === "agent" ? "Written by a language model from the tool results; every number was checked against them." : "Template report built from the tool results."} Click a number to see where it was measured.</p>
            </details>
          </Section>

          <Section title="Evidence">
            <ul className="space-y-1 text-[12px] text-ink-dim">{(report.evidence || []).map((e) => <li key={e.tool}><span aria-hidden="true" className={e.ok ? "text-accent" : "text-failure"}>{e.ok ? "✓" : "✗"}</span> <code className="num">{e.tool}</code> {e.ok ? "" : e.error}</li>)}</ul>
            <p className="mt-2 text-[12px] text-ink-dim">{report.ok ? "Number check: every number in the report matches a tool result and carries its unit." : `Number check found problems: ${JSON.stringify([...(report.unsupported_numbers || []), ...(report.missing_units || [])])}`} <a className="focus-ring rounded text-accent hover:underline" href={`/api/scans/${report.scan_id}/summary`} target="_blank" rel="noopener noreferrer">Raw tool results (JSON)</a></p>
          </Section>
        </motion.div>
      ) : null}
    </div>
  );
}
