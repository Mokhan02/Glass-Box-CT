import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { reducedMotion } from "../lib/motion";

const STEPS = [["segment_ct", "Segment the CT"], ["body_composition", "Measure muscle at L3"], ["measure_lesions", "Measure tumours"], ["quality_audit", "Audit the segmentation"]];

function Tick({ ok }) {
  return ok ? (
    <motion.svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><motion.path d="M3.5 8.5l3 3 6-7" stroke="#5EE6D0" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" initial={{ pathLength: reducedMotion() ? 1 : 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.28, ease: "easeOut" }} /></motion.svg>
  ) : (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="#FF7A6B" strokeWidth="1.8" strokeLinecap="round" /></svg>
  );
}

/** The agent's tool calls, replayed from the real evidence record: each step appears, runs, then ticks off. */
export default function ToolSteps({ evidence, done, onFinished }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!done) return;
    if (reducedMotion()) { setN(STEPS.length); onFinished && onFinished(); return; }
    setN(0); let k = 0; const t = setInterval(() => { k += 1; setN(k); if (k >= STEPS.length) { clearInterval(t); setTimeout(() => onFinished && onFinished(), 250); } }, 380);
    return () => clearInterval(t);
  }, [done]);
  const byTool = Object.fromEntries((evidence || []).map((e) => [e.tool, e]));
  return (
    <ol className="space-y-2" aria-label="Agent tool calls" aria-live="polite">
      {STEPS.map(([tool, label], i) => {
        const finished = done && i < n, running = !finished && (done ? i === n : i === 0), rec = byTool[tool];
        return (
          <li key={tool} className="flex items-center gap-3 text-[13px]">
            <span className="grid h-5 w-5 place-items-center">
              {finished ? <Tick ok={!rec || rec.ok} /> : running ? <span className="h-2 w-2 animate-pulse rounded-full bg-accent" aria-hidden="true" /> : <span className="h-1.5 w-1.5 rounded-full bg-white/20" aria-hidden="true" />}
            </span>
            <span className={finished ? "text-ink" : running ? "text-ink" : "text-ink-dim"}>{label}</span>
            <code className="num ml-auto text-[11px] text-ink-dim">{tool}</code>
            <span className="sr-only">{finished ? (rec && !rec.ok ? "failed" : "done") : running ? "running" : "waiting"}</span>
          </li>
        );
      })}
    </ol>
  );
}
