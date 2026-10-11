import { motion } from "framer-motion";
import { EASE } from "../lib/motion";
import { minus } from "../lib/format";

const num = (s) => parseFloat(String(s).replace("+", ""));

/** Horizontal estimate bars with 95% CI whiskers on a shared axis. Animates in when scrolled into view. Values are strings from findings.json (never recomputed). */
export default function RangeBars({ rows, domain, zero = true, unit = "", ariaLabel }) {
  const [lo, hi] = domain, x = (v) => `${(100 * (v - lo) / (hi - lo)).toFixed(2)}%`;
  return (
    <div role="img" aria-label={ariaLabel} className="space-y-3.5">
      {rows.map((r) => {
        const v = num(r.value), a = r.ci ? num(r.ci[0]) : null, b = r.ci ? num(r.ci[1]) : null, z = zero ? Math.max(lo, Math.min(hi, 0)) : lo;
        const left = Math.min(v, z), width = Math.abs(v - z);
        return (
          <div key={r.label} className="grid grid-cols-[minmax(0,9.5rem)_1fr_auto] items-center gap-3 sm:grid-cols-[12rem_1fr_auto]">
            <div className="text-[13px] leading-tight text-ink-dim">{r.label}</div>
            <div className="relative h-6">
              <div className="absolute inset-x-0 top-1/2 h-px bg-white/10" />
              {zero ? <div className="absolute inset-y-0 w-px bg-white/25" style={{ left: x(z) }} /> : null}
              <motion.div className="absolute top-0 h-full" style={{ left: x(left), width: `${(100 * width / (hi - lo)).toFixed(2)}%`, transformOrigin: v < z ? "right center" : "left center" }} initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true, margin: "-10% 0px" }} transition={{ duration: 0.7, ease: EASE }}>
                <div className="absolute inset-x-0 top-1/2 h-2.5 -translate-y-1/2 rounded-full" style={{ background: r.color || "#5EE6D0", minWidth: 3 }} />
              </motion.div>
              {r.ci ? <motion.div className="absolute top-1/2 h-px -translate-y-1/2 bg-ink" style={{ left: x(a), width: `${(100 * (b - a) / (hi - lo)).toFixed(2)}%` }} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.5, duration: 0.3 }} /> : null}
            </div>
            <div className="num text-right text-[13px] text-ink">{minus(r.value)}{unit}{r.ci ? <span className="block text-[11px] text-ink-dim">[{minus(r.ci[0])}, {minus(r.ci[1])}]</span> : null}</div>
          </div>
        );
      })}
    </div>
  );
}
