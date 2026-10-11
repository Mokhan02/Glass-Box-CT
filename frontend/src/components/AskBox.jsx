import { useRef, useState } from "react";
import { motion } from "framer-motion";
import ReportText from "./ReportText";
import { EASE } from "../lib/motion";

const LABEL = { segment_ct: "Segmentation", body_composition: "Body composition", measure_lesions: "Lesion measurements", quality_audit: "Quality audit" };
const EXAMPLES = ["Why was this flagged or not flagged?", "How reliable is the sum of diameters?", "Could a tumour be missed here?"];

/** "Ask about this scan": the question goes to the agent with this scan's tool results; the tool steps stream in, then the answer (numbers already checked server-side). */
export default function AskBox({ scanId, summary, onJump, available }) {
  const [q, setQ] = useState(""), [steps, setSteps] = useState([]), [ans, setAns] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const abort = useRef(null);
  const send = async (text) => {
    const question = (text ?? q).trim(); if (!question || busy) return;
    setQ(question); setBusy(true); setErr(""); setAns(null); setSteps([]); abort.current = new AbortController();
    try {
      const r = await fetch(`/api/scans/${scanId}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }), signal: abort.current.signal });
      if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch (e) { /* status text */ } throw new Error(m); }
      const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { done, value } = await reader.read(); if (done) break; buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === "step") setSteps((s) => { const rest = s.filter((x) => x.tool !== ev.tool); return [...rest, ev]; }); else if (ev.type === "answer") setAns(ev);
        }
      }
    } catch (e) { if (e.name !== "AbortError") setErr(e.message); } finally { setBusy(false); }
  };
  if (!available) return <p className="text-[13px] leading-relaxed text-ink-dim">Asking questions needs a language model, and none is configured on this server.</p>;
  return (
    <div>
      <form onSubmit={(e) => { e.preventDefault(); send(); }} className="flex gap-2">
        <label className="sr-only" htmlFor="ask-q">Ask a question about this scan</label>
        <input id="ask-q" value={q} onChange={(e) => setQ(e.target.value)} maxLength={300} placeholder="Ask about this scan's measurements or reliability…" className="focus-ring min-w-0 flex-1 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-[13px] text-ink placeholder:text-ink-dim" />
        <button type="submit" disabled={busy || !q.trim()} className="btn-pill !bg-accent !text-bg disabled:opacity-40" style={{ borderColor: "#5EE6D0" }}>{busy ? "Asking…" : "Ask"}</button>
      </form>
      <div className="mt-2.5 flex flex-wrap gap-1.5">{EXAMPLES.map((x) => <button key={x} type="button" className="btn-pill !py-1 !text-[12px]" disabled={busy} onClick={() => send(x)}>{x}</button>)}</div>
      {steps.length ? (
        <ol className="mt-4 space-y-1.5" aria-label="Tool calls for this question" aria-live="polite">
          {steps.map((s) => <li key={s.tool} className="flex items-center gap-2.5 text-[12px] text-ink-dim"><span aria-hidden="true" className={s.status === "running" ? "h-2 w-2 animate-pulse rounded-full bg-accent" : s.status === "done" ? "text-accent" : "text-failure"}>{s.status === "running" ? "" : s.status === "done" ? "✓" : "✗"}</span>{LABEL[s.tool] || s.tool}<code className="num ml-auto text-[11px]">{s.tool}</code></li>)}
        </ol>
      ) : null}
      {busy && !ans ? <p className="mt-3 text-[12px] text-ink-dim" role="status">Reading the tool results…</p> : null}
      {err ? <p role="alert" className="mt-3 text-[13px] text-failure">{err}</p> : null}
      {ans ? (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE }} className={`mt-4 rounded-xl border p-4 ${ans.refused ? "border-white/15 bg-white/[0.02]" : ans.ok ? "border-accent/30 bg-accent/[0.04]" : "border-caution/30 bg-caution/[0.05]"}`} aria-live="polite">
          <div className="eyebrow mb-2">{ans.refused ? "Not answered" : ans.ok ? "Answer" : "Could not answer"}</div>
          {ans.ok && !ans.refused ? <ReportText text={ans.answer} summary={summary} onJump={onJump} /> : <p className="text-[13px] leading-relaxed text-ink-dim">{ans.answer}</p>}
          {ans.ok && !ans.refused ? <p className="mt-1 text-[11px] text-ink-dim">Every number was checked against this scan's tool results ({(ans.tools || []).join(", ")}).</p> : null}
        </motion.div>
      ) : null}
    </div>
  );
}
