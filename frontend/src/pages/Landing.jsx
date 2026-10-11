import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { listScans, fileUrl, getHealth, getJSON } from "../lib/api";
import { EASE } from "../lib/motion";
import Chip from "../components/Chip";

const CHIP = { good: "good", flagged: "flagged", missed: "missed" };

function ScanCard({ s, i }) {
  const kind = s.verdict || "review";
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

function Upload({ live }) {
  const nav = useNavigate(), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const send = async (e) => {
    const f = e.target.files[0]; if (!f) return; setBusy(true); setErr(""); const fd = new FormData(); fd.append("file", f);
    try { const r = await getJSON("/api/segment", { method: "POST", body: fd }); nav(`/scan/${r.scan_id}`); } catch (x) { setErr(x.message); setBusy(false); }
  };
  return (
    <div className={`glass mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 p-4 sm:p-5 ${live ? "" : "opacity-60"}`}>
      <div className="min-w-[14rem] flex-1"><div className="eyebrow">Try your own scan {live ? null : <span className="ml-1.5 rounded-full border border-white/20 px-2 py-0.5 normal-case tracking-normal">coming soon</span>}</div>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-dim">{live ? "Upload a public or synthetic research CT (NIfTI). Never upload patient data." : "Segmenting your own scan needs a GPU backend, which this server does not have yet."}</p></div>
      <label className={`btn-pill ${live ? "cursor-pointer" : "pointer-events-none cursor-not-allowed"}`} aria-disabled={!live}>{busy ? "Segmenting…" : "Choose a NIfTI file"}<input type="file" accept=".nii,.gz" className="sr-only" disabled={!live || busy} onChange={send} /></label>
      {err ? <p role="alert" className="w-full text-[13px] text-failure">{err}</p> : null}
    </div>
  );
}

export default function Landing() {
  const [data, setData] = useState(null), [err, setErr] = useState(""), [live, setLive] = useState(false); const scans = data && data.scans;
  useEffect(() => { listScans().then(setData).catch((e) => setErr(e.message)); getHealth().then((h) => setLive(!!h.live_segmentation)).catch(() => {}); }, []);
  return (
    <div className="mx-auto max-w-[1500px] px-4 pt-12 sm:px-6 sm:pt-20">
      <section className="max-w-3xl">
        <div className="eyebrow">A trust layer for oncology imaging AI</div>
        <h1 className="mt-3 text-[34px] font-medium leading-[1.08] tracking-tight sm:text-[56px]">Don't just measure the tumour. <span className="text-ink-dim">Measure whether to trust the measurement.</span></h1>
        <p className="mt-5 max-w-2xl text-[16px] leading-relaxed text-ink-dim">Glass-Box CT audits an AI model's lesion measurements and gives each scan and each lesion a verdict, Trusted, Review or Unreliable, with the reasons and the evidence behind them. Five held-out research scans of an open NVIDIA model below.</p>
        <div className="mt-6 flex flex-wrap gap-2">
          <a href="#scans" className="btn-pill" style={{ background: "#5EE6D0", color: "#07090D", borderColor: "#5EE6D0" }}>Explore the scans</a>
          <Link to="/evidence" className="btn-pill">The evidence</Link>
        </div>
      </section>
      <section id="scans" className="mt-14 scroll-mt-24" aria-labelledby="scans-h">
        <h2 id="scans-h" className="eyebrow">Five scans, chosen by rule, not at random</h2>
        {err ? <p role="alert" className="mt-4 text-failure">Could not load the scans: {err}</p> : null}
        {scans && !scans.length ? <p className="mt-4 text-ink-dim">No sample scans are installed on this server.</p> : null}
        <ul className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {(scans || []).map((s, i) => <ScanCard key={s.id} s={s} i={i} />)}
        </ul>
        {scans ? <Upload live={live} /> : null}
        {scans ? <p className="mt-4 max-w-2xl text-[12px] leading-relaxed text-ink-dim">{data.selection_note}</p> : null}
      </section>
    </div>
  );
}
