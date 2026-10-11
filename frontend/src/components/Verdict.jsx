import { motion } from "framer-motion";
import { EASE } from "../lib/motion";

const PLAIN = {
  "predicted tumour volume": "the size of the predicted tumour",
  "mean model probability inside the predicted tumour": "how confident the model is inside its predicted tumour",
  "mean prediction entropy inside the predicted tumour": "how uncertain the model's voxel decisions are inside it",
};
const KIND = { bad: ["#FF7A6B", "rgba(255,122,107,0.08)", "Flagged"], good: ["#5EE6D0", "rgba(94,230,208,0.07)", "No flag"], na: ["#F5B54A", "rgba(245,181,74,0.08)", "Cannot assess"] };

/** One plain sentence and the flag colour, then one plain-language reason (no bars). Colour is always paired with the words. */
export function verdictOf(q) {
  if (!q) return { kind: "na", sentence: "No quality assessment is available for this scan", reason: "" };
  if (!q.assessed) return { kind: "na", sentence: "No tumour predicted — that does not rule out a tumour", reason: q.reason || "" };
  const bad = q.flag === "likely wrong", dir = bad ? "failure" : "success";
  const ds = (q.drivers || []).filter((d) => d.pushes_toward === dir).slice(0, 2).map((d) => PLAIN[d.name] || d.name);
  const why = ds.length ? `${ds.join(" and ")} look${ds.length === 1 ? "s" : ""} like cases where this model got the tumour ${bad ? "wrong" : "right"}.` : (bad ? "The signals look like cases where this model got the tumour wrong." : "The signals look like cases where this model got the tumour right.");
  return { kind: bad ? "bad" : "good", sentence: bad ? "Tumour likely under-segmented — don't trust these measurements" : "Segmentation looks reliable for this case", reason: `Estimated probability the segmentation is wrong: ${q.failure_probability}. ${why[0].toUpperCase() + why.slice(1)}` };
}

export default function Verdict({ q }) {
  const v = verdictOf(q), [c, bg, label] = KIND[v.kind];
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE }} className="rounded-2xl border p-4 sm:p-5" style={{ borderColor: `${c}55`, background: bg }} role="status">
      <div className="flex items-center gap-2 text-[12px] font-medium" style={{ color: c }}><span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: c }} />{label}</div>
      <p className="mt-2 text-[21px] font-medium leading-snug tracking-tight text-ink">{v.sentence}</p>
      {v.reason ? <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">{v.reason}</p> : null}
    </motion.div>
  );
}
