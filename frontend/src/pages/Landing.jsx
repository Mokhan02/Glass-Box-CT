import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { listScans, fileUrl } from "../lib/api";
import { EASE } from "../lib/motion";
import Chip from "../components/Chip";

const CHIP = { good: "good", flagged: "flagged", missed: "missed" };

function ScanCard({ s, i }) {
  const kind = CHIP[s.chip] || "flagged";
  return (
    <motion.li initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE, delay: 0.05 * i }}>
      <Link to={`/scan/${s.id}`} className="glass focus-ring group block h-full overflow-hidden transition-colors duration-300 hover:border-white/20">
        <div className="relative aspect-[4/3] overflow-hidden bg-black">
          <img src={fileUrl(s.id, "thumb.png")} alt={`Axial CT slice of scan ${s.id.replace("demo_", "")} with the model's predicted tumour outlined`} loading="lazy" className="h-full w-full object-cover opacity-90 transition-transform duration-700 ease-out group-hover:scale-[1.03]" />
          <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/70 to-transparent" />
          <div className="absolute left-3 top-3"><Chip kind={kind} /></div>
        </div>
        <div className="p-4">
          <div className="eyebrow">Scan {s.id.replace("demo_", "")}</div>
          <p className="mt-1.5 text-[15px] leading-snug text-ink">{s.oneliner || s.title}</p>
          <span className="mt-3 inline-flex items-center gap-1 text-[13px] text-accent">Open <span aria-hidden="true" className="transition-transform duration-300 group-hover:translate-x-0.5">→</span></span>
        </div>
      </Link>
    </motion.li>
  );
}

export default function Landing() {
  const [data, setData] = useState(null), [err, setErr] = useState(""); const scans = data && data.scans;
  useEffect(() => { listScans().then(setData).catch((e) => setErr(e.message)); }, []);
  return (
    <div className="mx-auto max-w-[1500px] px-4 pt-12 sm:px-6 sm:pt-20">
      <section className="max-w-3xl">
        <div className="eyebrow">Research instrument · liver tumours on CT</div>
        <h1 className="mt-3 text-[34px] font-medium leading-[1.08] tracking-tight sm:text-[56px]">See where the model fails <span className="text-ink-dim">— and why.</span></h1>
        <p className="mt-5 max-w-2xl text-[16px] leading-relaxed text-ink-dim">An AI segmentation model measures tumours on CT. This tool shows the measurement, flags when it should not be trusted, and opens the model to show what goes wrong. Five held-out research scans below.</p>
        <div className="mt-6 flex flex-wrap gap-2">
          <a href="#scans" className="btn-pill" style={{ background: "#5EE6D0", color: "#07090D", borderColor: "#5EE6D0" }}>Explore the scans</a>
          <Link to="/why" className="btn-pill">Why it fails</Link>
        </div>
      </section>
      <section id="scans" className="mt-14 scroll-mt-24" aria-labelledby="scans-h">
        <h2 id="scans-h" className="eyebrow">Five scans, chosen by rule, not at random</h2>
        {err ? <p role="alert" className="mt-4 text-failure">Could not load the scans: {err}</p> : null}
        {scans && !scans.length ? <p className="mt-4 text-ink-dim">No sample scans are installed on this server.</p> : null}
        <ul className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {(scans || []).map((s, i) => <ScanCard key={s.id} s={s} i={i} />)}
        </ul>
        {scans ? <p className="mt-4 max-w-2xl text-[12px] leading-relaxed text-ink-dim">{data.selection_note}</p> : null}
      </section>
    </div>
  );
}
