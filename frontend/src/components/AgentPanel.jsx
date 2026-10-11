import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Measure from "./Measure";
import ToolSteps from "./ToolSteps";
import RecalToggle from "./RecalToggle";
import ReportText, { stripShownElsewhere } from "./ReportText";
import Verdict from "./Verdict";
import LesionsTable from "./LesionsTable";
import Collapsible from "./Collapsible";
import ScanSwitcher from "./ScanSwitcher";
import AskBox from "./AskBox";
import { EASE } from "../lib/motion";
import { minus } from "../lib/format";

const fade = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.25, ease: EASE } };
const H = ({ children }) => <h3 className="eyebrow mb-3">{children}</h3>;

export default function AgentPanel({ meta, all, summary, report, error, mode, setMode, onJump, canAsk }) {
  const [finished, setFinished] = useState(false);
  const les = (summary && summary.lesions && summary.lesions.lesions) || [];
  const q = summary && summary.quality_audit && summary.quality_audit.structures && summary.quality_audit.structures[0];
  const bc = summary && summary.body_composition, L = (summary && summary.lesions) || {};
  const sorted = useMemo(() => [...les].sort((a, b) => b.longest_axial_diameter.value - a.longest_axial_diameter.value), [les]);
  const targets = sorted.filter((l) => l.recist === "target candidate"), sum = L.sum_of_target_diameters ? L.sum_of_target_diameters.value : targets.reduce((a, l) => a + l.longest_axial_diameter.value, 0);
  const contrast = L.tumour_liver_contrast ? L.tumour_liver_contrast.value : null;
  const refC = meta && Number.isFinite(meta.reference_contrast_hu) ? meta.reference_contrast_hu : null, cov = meta && meta.reference_coverage_pct;
  const why = contrast !== null && refC !== null && Math.abs(contrast - refC) >= 5 && cov < 50 ? `The model found only ${cov}% of this tumour, ${contrast < refC ? "the darker" : "the brighter"} part, so the contrast measured on its prediction differs from the reference tumour's.` : null;
  const agent = report && report.mode === "agent", v = L.validation_context;
  const prose = report ? stripShownElsewhere(report.report, summary) : "";
  return (
    <div>
      <ScanSwitcher scans={all} current={meta && meta.id} />
      <div className="mt-5 flex items-baseline justify-between"><h2 className="text-[17px] font-medium tracking-tight">Agent</h2><span className="text-[12px] text-ink-dim">{finished ? "4 tool calls · see Evidence" : "working…"}</span></div>
      <AnimatePresence initial={false}>
        {!finished ? <motion.div key="steps" exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.25 }} className="mt-4 overflow-hidden"><ToolSteps evidence={report && report.evidence} done={!!report} fast={!!(report && report.cached)} onFinished={() => setFinished(true)} /></motion.div> : null}
      </AnimatePresence>
      {error ? <p role="alert" className="mt-4 text-failure">{error}</p> : null}

      {report && finished ? (
        <motion.div {...fade} className="mt-4">
          <Verdict trust={summary.trust} />

          <section className="py-6" aria-label="Key numbers">
            {sorted.length ? (
              <div className="grid grid-cols-3 gap-x-4">
                <Measure label="Largest lesion" value={sorted[0].longest_axial_diameter.value} unit="mm" size="text-[30px]" />
                <Measure label="Sum of diameters" value={sum} unit="mm" size="text-[30px]" hint={`${targets.length} target${targets.length === 1 ? "" : "s"}`} />
                {contrast !== null ? <Measure label="Tumour − liver" value={contrast} unit="HU" size="text-[30px]" hint={refC !== null ? `reference ${refC > 0 ? "+" : ""}${minus(String(refC))} HU` : "on the prediction"} /> : <div />}
              </div>
            ) : <p className="text-[13px] leading-relaxed text-ink-dim">{L.note || "No tumour predicted."}</p>}
            {why ? <p className="mt-3 text-[12px] leading-relaxed text-ink-dim">{why}</p> : null}
          </section>

          {summary.phase && summary.phase.estimate ? (
            <section className="border-t border-white/[0.07] py-5" aria-label="Contrast phase"><H>Contrast phase <span className="ml-1 normal-case tracking-normal text-caution">research heuristic</span></H>
              <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
                <span className="text-[20px] font-medium capitalize">{summary.phase.estimate}</span>
                <span className="text-[13px] text-ink-dim">aorta {summary.phase.aorta_hu ? <><span className="num text-ink">{summary.phase.aorta_hu.value}</span> <span className="unit">HU</span></> : "not measurable"} · portal vein {summary.phase.portal_vein_hu ? <><span className="num text-ink">{summary.phase.portal_vein_hu.value}</span> <span className="unit">HU</span></> : "not measurable"}</span>
              </div>
              <p className="mt-2 text-[12px] leading-relaxed text-ink-dim">{summary.phase.note} Estimated from the model's own aorta and portal-vein masks with literature thresholds; not a validated classifier.</p>
            </section>
          ) : null}

          <section className="border-t border-white/[0.07] py-5" aria-label="Threshold"><H>Threshold</H><RecalToggle recal={summary.recalibrated} mode={mode} setMode={setMode} /></section>

          {sorted.length ? <section className="border-t border-white/[0.07] py-5" aria-label="Lesions"><H>Lesions</H><LesionsTable lesions={sorted} trust={summary.trust} onJump={onJump} /></section> : null}

          {prose ? (agent ? (
            <section className="border-t border-white/[0.07] py-5" aria-label="Report"><H>Report</H><ReportText text={prose} summary={summary} onJump={onJump} /></section>
          ) : (
            <Collapsible title="Written report"><ReportText text={prose} summary={summary} onJump={onJump} /></Collapsible>
          )) : null}

          <section className="border-t border-white/[0.07] py-5" aria-label="Ask about this scan"><H>Ask about this scan</H><AskBox scanId={meta ? meta.id : report.scan_id} summary={summary} onJump={onJump} available={canAsk} /></section>

          <Collapsible title="Known limits of this model" open={!!L.contrast_caution}>
            <div className="space-y-3 text-[13px] leading-relaxed text-ink-dim">
              {L.standing_limitation ? <p className="text-ink">{L.standing_limitation}</p> : null}
              {v ? <p>On held-out research data the model measured detected lesions a median <span className="num text-ink">{minus(String(v.detected_lesion_median_error.value))} mm</span> (<span className="num text-ink">{minus(String(v.detected_lesion_median_relative_error.value))}%</span>) from the reference and the sum of diameters a median <span className="num text-ink">{minus(String(v.case_sum_of_diameters_median_error.value))}%</span>; it never overestimated by <span className="num text-ink">{v.large_overestimate_threshold.value}%</span> or more. {v.note}</p> : null}
              {L.contrast_caution ? <p className="rounded-lg border border-caution/30 bg-caution/[0.06] p-3 text-ink">{L.contrast_caution}<span className="mt-1.5 block text-[12px] text-ink-dim">{L.contrast_context && L.contrast_context.evidence_status}</span></p> : null}
            </div>
          </Collapsible>

          {bc ? (
            <Collapsible title="Body composition">
              {bc.available ? <div className="space-y-2 text-[13px] text-ink-dim"><p>Psoas and paraspinal muscle at L3: <span className="num text-ink">{bc.psoas_paraspinal_area.value}</span> <span className="unit">cm2</span> · psoas alone <span className="num text-ink">{bc.psoas_area.value}</span> <span className="unit">cm2</span> · mean density <span className="num text-ink">{bc.mean_muscle_density.value}</span> <span className="unit">HU</span></p><p className="text-[12px]">{bc.label}</p></div> : <p className="text-[13px] text-ink-dim">Not available: {bc.reason}</p>}
            </Collapsible>
          ) : null}

          <section className="border-t border-white/[0.07] py-5" aria-label="Audit record"><H>Audit record</H>
            <p className="text-[13px] leading-relaxed text-ink-dim">Everything behind this verdict in one file: model and checkpoint hash, thresholds, phase estimate, flag score and reasons, default vs recalibrated values, the agent's report and the tool-call log, with an integrity checksum.</p>
            <div className="mt-3 flex flex-wrap gap-2"><a className="btn-pill" href={`/api/scans/${meta ? meta.id : report.scan_id}/audit.pdf`} download>Download PDF</a><a className="btn-pill" href={`/api/scans/${meta ? meta.id : report.scan_id}/audit.json`} download>Download JSON</a></div>
          </section>

          <Collapsible title="Evidence">
            <ul className="space-y-1 text-[12px] text-ink-dim">{(report.evidence || []).map((e) => <li key={e.tool}><span aria-hidden="true" className={e.ok ? "text-accent" : "text-failure"}>{e.ok ? "✓" : "✗"}</span> <code className="num">{e.tool}</code> {e.ok ? "" : e.error}</li>)}</ul>
            <p className="mt-2.5 text-[12px] leading-relaxed text-ink-dim">{report.ok ? "Number check: every number in the report matches a tool result and carries its unit." : `Number check found problems: ${JSON.stringify([...(report.unsupported_numbers || []), ...(report.missing_units || [])])}`}</p>
            <p className="mt-1.5 text-[12px] leading-relaxed text-ink-dim">Report source: {agent ? <>written by a language model (<span className="num">{report.provider_host}</span>{report.models ? <>, <span className="num">{report.models.reasoning}</span></> : null}) from the tool results.</> : <>template built from the tool results; no language model was called.</>}{report.note ? ` ${report.note}.` : ""} <a className="focus-ring rounded text-accent hover:underline" href={`/api/scans/${report.scan_id}/summary`} target="_blank" rel="noopener noreferrer">Raw tool results (JSON)</a></p>
          </Collapsible>
        </motion.div>
      ) : null}
    </div>
  );
}
