import { motion } from "framer-motion";
import CountUp from "./CountUp";
import findings from "../data/findings.json";
import { minus } from "../lib/format";

/** Default vs recalibrated (research): tumour measurements and sum of diameters under both thresholds, with the cost stated. Default stays primary. */
export default function RecalToggle({ recal, mode, setMode, flagged }) {
  const f = findings.bias, g = findings.gated;
  if (!recal) {
    return <p className="text-[12px] leading-relaxed text-ink-dim">The recalibrated (research) view is not available for this scan.</p>;
  }
  const d = recal.default, r = recal.recalibrated, max = Math.max(d.sum_of_diameters.value, r.sum_of_diameters.value, 1) * 1.12, showRecal = mode === "recal" || (mode === "gated" && flagged), cur = showRecal ? r : d;
  const pct = (v) => `${(100 * v / max).toFixed(1)}%`;
  return (
    <div>
      <div role="tablist" aria-label="Threshold" className="inline-flex gap-1 rounded-full border border-white/10 p-1">
        {[["default", "Default"], ["recal", "Recalibrated (research)"], ["gated", "Flag-gated (research)"]].map(([k, label]) => (
          <button key={k} role="tab" aria-selected={mode === k} onClick={() => setMode(k)} className="btn-pill !border-transparent !py-1">{label}</button>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div><div className="eyebrow">Sum of diameters</div><div className="mt-1 text-[28px] leading-none"><CountUp value={cur.sum_of_diameters.value} /> <span className="unit text-[14px]">mm</span></div></div>
        <div><div className="eyebrow">Largest lesion</div><div className="mt-1 text-[28px] leading-none"><CountUp value={cur.largest_diameter.value} /> <span className="unit text-[14px]">mm</span></div></div>
      </div>
      <div className="relative mt-4 h-2.5 rounded-full bg-white/[0.06]" role="img" aria-label={`Sum of diameters: default ${d.sum_of_diameters.value} mm, recalibrated ${r.sum_of_diameters.value} mm`}>
        <motion.div className="absolute inset-y-0 left-0 rounded-full" style={{ background: showRecal ? "#F5B54A" : "#5EE6D0" }} initial={false} animate={{ width: pct(cur.sum_of_diameters.value) }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} />
        <span className="absolute -top-1 h-4.5 w-px bg-white/50" style={{ left: pct(showRecal ? d.sum_of_diameters.value : r.sum_of_diameters.value), height: 18 }} aria-hidden="true" />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-ink-dim"><span>0 mm</span><span>marker: {showRecal ? "default" : "recalibrated"} ({(showRecal ? d : r).sum_of_diameters.value} mm)</span></div>
      {mode === "default" ? <p className="mt-3 text-[12px] leading-relaxed text-ink-dim">The default threshold (probability above 0.5) is the primary measurement. It tends to under-measure; the other two options show what a lower threshold would report.</p> : null}
      {mode === "recal" ? (
        <p className="mt-3 rounded-lg border border-caution/30 bg-caution/[0.06] px-3 py-2 text-[12px] leading-relaxed text-ink">
          <strong className="font-medium text-caution">The cost.</strong> Recalibration lowers the tumour threshold (logit {minus(String(recal.threshold_logit))}). On held-out research data it moved the median sum-of-diameters error from <span className="num">{minus(f.orig.sld)}%</span> to <span className="num">{minus(f.recal.sld)}%</span>, but <strong className="font-medium">overestimates by ≥20% in <span className="num">{f.recal.over}%</span> of cases</strong>. The default stays primary.
        </p>
      ) : null}
      {mode === "gated" ? (
        <p className="mt-3 rounded-lg border border-accent/30 bg-accent/[0.05] px-3 py-2 text-[12px] leading-relaxed text-ink">
          <strong className="font-medium text-accent">Flag-gated.</strong> {flagged ? "The quality flag says this segmentation is likely wrong, so the recalibrated threshold is used for this scan." : "The quality flag is quiet for this scan, so the default threshold is kept."} In cross-validation on <span className="num">{g.n.value}</span> held-out scans this rule gave a mean absolute sum-of-diameters error of <span className="num">{g.abs.value}%</span> (default <span className="num">{g.default_abs.value}%</span>) and over-estimated by ≥20% in <span className="num">{g.over.value}%</span> of cases (always-recalibrated <span className="num">{g.recal_over.value}%</span>), with a median error of <span className="num">{minus(g.median.value)}%</span>. The thresholds were chosen on these same scans, so expect less on new data.
        </p>
      ) : null}
    </div>
  );
}
