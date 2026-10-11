import { motion } from "framer-motion";
import { EASE } from "../lib/motion";

const SENTENCE = { Trusted: "This measurement looks reliable for this case", Review: "Review this measurement before relying on it", Unreliable: "Don't rely on this measurement" };
const KIND = { Trusted: ["#5EE6D0", "rgba(94,230,208,0.07)"], Review: ["#F5B54A", "rgba(245,181,74,0.08)"], Unreliable: ["#FF7A6B", "rgba(255,122,107,0.08)"] };

/** The scan verdict: Trusted / Review / Unreliable, one plain sentence, and the reasons (each from a named tool result). Colour is always paired with the word. */
export default function Verdict({ trust }) {
  if (!trust) return null;
  const [c, bg] = KIND[trust.level] || KIND.Review;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE }} className="rounded-2xl border p-4 sm:p-5" style={{ borderColor: `${c}55`, background: bg }} role="status">
      <div className="flex items-center gap-2 text-[12px] font-medium uppercase tracking-[0.12em]" style={{ color: c }}><span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: c }} />{trust.level}</div>
      <p className="mt-2 text-[21px] font-medium leading-snug tracking-tight text-ink">{SENTENCE[trust.level]}</p>
      <ul className="mt-3 space-y-1.5">
        {trust.reasons.map((r) => <li key={r.code} title={`Source: ${r.source}`} className="flex gap-2 text-[13px] leading-relaxed text-ink-dim"><span aria-hidden="true" className="mt-[7px] h-1 w-1 shrink-0 rounded-full" style={{ background: c }} />{r.text}</li>)}
      </ul>
      <p className="mt-3 text-[11px] text-ink-dim">Research heuristic about the measurement, not a diagnosis.</p>
    </motion.div>
  );
}
