import { motion } from "framer-motion";
import findings from "../data/findings.json";
import { EASE } from "../lib/motion";
import { minus } from "../lib/format";

/** Shown only when the quality flag says the segmentation is likely wrong (the gate fires). Default stays the primary measurement; this is a suggestion with its value, its caveat
 *  and the over-estimation rate beside it. The cross-validated numbers are from the nested analysis, the more cautious of the two (results/SUMMARY.md section 7, results/trust/DEVIATIONS.md #1). */
export default function Suggestion({ recal, flagged, mode, setMode }) {
  const g = findings.gated;
  if (!recal || !flagged) return null;
  const d = recal.default, r = recal.recalibrated, on = mode === "recal";
  return (
    <motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE }} aria-label="Trust layer suggestion" className="rounded-2xl border border-accent/35 bg-accent/[0.05] p-4 sm:p-5">
      <div className="eyebrow !text-accent">Trust layer suggestion (research): flag-gated recalibration</div>
      <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">The quality flag says this segmentation is likely wrong, and the default tends to under-measure in that situation. A lower threshold would report:</p>
      <div className="mt-3 grid grid-cols-3 gap-4">
        <div><div className="eyebrow">Sum of diameters</div><div className="mt-1 text-[26px] leading-none"><span className="num">{r.sum_of_diameters.value}</span> <span className="unit text-[13px]">mm</span></div><div className="mt-1 text-[11px] text-ink-dim">default {d.sum_of_diameters.value} mm</div></div>
        <div><div className="eyebrow">Largest lesion</div><div className="mt-1 text-[26px] leading-none"><span className="num">{r.largest_diameter.value}</span> <span className="unit text-[13px]">mm</span></div><div className="mt-1 text-[11px] text-ink-dim">default {d.largest_diameter.value} mm</div></div>
        <div><div className="eyebrow">Over-estimates ≥20%</div><div className="mt-1 text-[26px] leading-none text-caution"><span className="num">{g.over.value}</span><span className="unit text-[13px]">%</span></div><div className="mt-1 text-[11px] text-ink-dim">of cases (default {g.default_over.value}%, always-recalibrated {g.recal_over.value}%)</div></div>
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-dim">Caveat: in a nested cross-validation (thresholds chosen on training folds only) on <span className="num text-ink">{g.n.value}</span> held-out scans this rule gave a mean absolute sum-of-diameters error of <span className="num text-ink">{g.abs.value}%</span> (default <span className="num text-ink">{g.default_abs.value}%</span>) and a median error of <span className="num text-ink">{minus(g.median.value)}%</span>. Over-estimation of ≥20% was {g.prereg_over.value}% in the pre-registered analysis, and the nested figure shown here is the more cautious one; its 95% interval is {g.over.ci[0]} to {g.over.ci[1]}%, so it is not clearly better than always recalibrating.</p>
      <button className="btn-pill mt-3" aria-pressed={on} onClick={() => setMode(on ? "default" : "recal")}>{on ? "Back to the default measurement" : "Show on the scan"}</button>
    </motion.section>
  );
}
