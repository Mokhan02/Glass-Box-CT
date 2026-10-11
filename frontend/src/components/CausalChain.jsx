import { motion } from "framer-motion";
import { EASE } from "../lib/motion";
import f from "../data/findings.json";
import { minus } from "../lib/format";

const TAG = { pre: ["Pre-registered", "#5EE6D0"], expl: ["Exploratory", "#F5B54A"] };

/** The causal chain, ending the story. Each link names the experiment that supports it and whether it was pre-registered. */
export default function CausalChain() {
  const nodes = [
    { t: "A brighter tumour", d: "Edit a tumour from dark to bright and nothing else changes.", tag: "pre", ev: `Coverage ${minus(f.brightness.armA.value)} [${minus(f.brightness.armA.ci[0])}, ${minus(f.brightness.armA.ci[1])}] — ${f.brightness.armA.hits} tumours` },
    { t: "Weaker tumour feature", d: "The tumour-vs-liver feature in the middle of the network drops inside that tumour.", tag: "expl", ev: `Lower in ${f.patching.bridge.hits} tumours` },
    { t: "Below the cut-off", d: "The final score sits under the decision threshold, so the tumour is missed.", tag: "pre", ev: `Removing the feature: lesion Dice ${minus(f.erasure.leace.value)} [${minus(f.erasure.leace.ci[0])}, ${minus(f.erasure.leace.ci[1])}]` },
    { t: "A threshold fixes most of it", d: "Lowering the threshold recovers tumours, at a stated cost.", tag: "pre", ev: `Over-estimates by ≥20% in ${f.bias.recal.over}% of cases` },
  ];
  return (
    <ol className="grid gap-3 md:grid-cols-4" aria-label="The causal chain">
      {nodes.map((n, i) => (
        <motion.li key={n.t} className="glass relative p-4" initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-8% 0px" }} transition={{ duration: 0.4, ease: EASE, delay: 0.12 * i }}>
          <div className="eyebrow" style={{ color: TAG[n.tag][1] }}>{TAG[n.tag][0]}</div>
          <div className="mt-1.5 text-[15px] font-medium leading-snug">{n.t}</div>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-dim">{n.d}</p>
          <p className="num mt-2.5 text-[12px] leading-snug text-ink">{n.ev}</p>
          {i < nodes.length - 1 ? <span aria-hidden="true" className="absolute -right-2.5 top-1/2 hidden -translate-y-1/2 text-accent md:block">→</span> : null}
        </motion.li>
      ))}
    </ol>
  );
}
