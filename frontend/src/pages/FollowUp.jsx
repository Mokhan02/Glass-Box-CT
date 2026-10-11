import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { getJSON } from "../lib/api";
import { EASE } from "../lib/motion";
import { minus } from "../lib/format";

const CATS = ["CR", "PR", "SD", "PD"], NAME = { CR: "Complete response", PR: "Partial response", SD: "Stable disease", PD: "Progression" };
const pct = (x) => `${Math.round(100 * x)}%`;

function Slider({ id, label, value, onChange, min, max, step = 1, unit, fmt = (v) => v }) {
  return (
    <div>
      <div className="flex items-baseline justify-between"><label htmlFor={id} className="text-[13px] text-ink-dim">{label}</label><span className="num text-[15px] text-ink">{fmt(value)}<span className="unit"> {unit}</span></span></div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="focus-ring mt-1.5 w-full accent-[#5EE6D0]" />
    </div>
  );
}
const Seg = ({ label, value, onChange }) => (
  <div><div className="text-[13px] text-ink-dim">{label}</div>
    <div role="group" aria-label={label} className="mt-1.5 inline-flex gap-1 rounded-full border border-white/10 p-1">{[[false, "Dark tumour"], [true, "Bright tumour"]].map(([v, t]) => <button key={t} className="btn-pill !border-transparent !py-1 !text-[12px]" aria-pressed={value === v} onClick={() => onChange(v)}>{t}</button>)}</div></div>
);

function Method({ title, m, truth, accent }) {
  return (
    <div className="glass p-4 sm:p-5">
      <div className="eyebrow" style={{ color: accent }}>{title}</div>
      <div className="mt-3 space-y-2" role="img" aria-label={`${title}: probability of each response category`}>
        {CATS.map((c) => (
          <div key={c} className="grid grid-cols-[2.2rem_1fr_3rem] items-center gap-2 text-[13px]">
            <span className={`num ${c === truth ? "text-ink" : "text-ink-dim"}`}>{c}</span>
            <div className="h-2.5 rounded-full bg-white/[0.06]"><motion.div className="h-full rounded-full" style={{ background: c === truth ? "#5EE6D0" : accent, opacity: c === truth ? 1 : 0.75 }} initial={false} animate={{ width: pct(m.category_probability[c]) }} transition={{ duration: 0.45, ease: EASE }} /></div>
            <span className="num text-right text-ink">{pct(m.category_probability[c])}</span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-dim">Mean measured sum: <span className="num text-ink">{m.mean_baseline_sum_mm.toFixed(1)}</span> to <span className="num text-ink">{m.mean_followup_sum_mm.toFixed(1)}</span> <span className="unit">mm</span>. Wrong category in <span className="num text-ink">{pct(m.wrong_probability)}</span> of simulated draws.</p>
    </div>
  );
}

export default function FollowUp() {
  const [model, setModel] = useState(null), [a, setA] = useState(55), [b, setB] = useState(30), [chg, setChg] = useState(-20), [bb, setBb] = useState(false), [bf, setBf] = useState(false), [res, setRes] = useState(null), [err, setErr] = useState("");
  const tick = useRef(0);
  useEffect(() => { getJSON("/api/longitudinal/model").then(setModel).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!model || !model.available) return; const n = ++tick.current;
    const t = setTimeout(() => getJSON("/api/longitudinal/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ baseline_mm: b >= 10 ? [a, b] : [a], change: chg / 100, bright_baseline: bb, bright_followup: bf }) }).then((r) => { if (n === tick.current) { setRes(r); setErr(""); } }).catch((e) => setErr(e.message)), 250);
    return () => clearTimeout(t);
  }, [model, a, b, chg, bb, bf]);
  const preset = (p) => { setA(p.baseline_mm[0]); setB(p.baseline_mm[1] || 0); setChg(Math.round(p.change * 100)); setBb(p.bright_baseline); setBf(p.bright_followup); };
  const cd = res && res.calibration_dependent;
  return (
    <div className="mx-auto max-w-[1200px] px-4 pt-10 sm:px-6 sm:pt-16">
      <div role="note" className="rounded-xl border border-caution/50 bg-caution/[0.08] px-4 py-3 text-[13px] leading-relaxed text-ink"><strong className="font-semibold text-caution">SIMULATED.</strong> Everything on this page is generated from a bias model fitted to real per-lesion measurements. These are not real patients, and no follow-up scans were used.{model && model.fixture ? <strong className="ml-1 text-failure"> TEST FIXTURE MODEL: made-up data.</strong> : null}</div>
      <div className="mt-8"><div className="eyebrow">Follow-up check</div>
        <h1 className="mt-2 text-[32px] font-medium leading-tight tracking-tight sm:text-[44px]">Does the response depend on the calibration?</h1>
        <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-dim">Cancer response is judged by how the sum of target diameters changes between two scans (RECIST 1.1). The model under-measures by an amount that depends on size and on how bright the tumour is, so the error does not cancel between visits. Pick a true change and see what each threshold would report.</p></div>
      {err && !model ? <p role="alert" className="mt-6 text-failure">{err}</p> : null}
      {model && !model.available ? <p className="mt-6 text-ink-dim">The two-timepoint bias model is not installed on this server.</p> : null}
      {model && model.available ? (
        <div className="mt-8 grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="glass space-y-5 self-start p-5">
            <div className="flex flex-wrap gap-1.5">{(model.presets || []).map((p) => <button key={p.label} className="btn-pill !py-1 !text-[12px]" onClick={() => preset(p)}>{p.label.replace("Simulated patient ", "Patient ").split(",")[0]}</button>)}</div>
            <Slider id="a" label="Target lesion 1 at baseline" value={a} onChange={setA} min={12} max={120} unit="mm" />
            <Slider id="b" label="Target lesion 2 at baseline (0 = none)" value={b} onChange={setB} min={0} max={120} unit="mm" fmt={(v) => (v < 10 ? "none" : v)} />
            <Slider id="chg" label="True change in the sum of diameters" value={chg} onChange={setChg} min={-60} max={60} unit="%" fmt={(v) => `${v > 0 ? "+" : ""}${minus(String(v))}`} />
            <Seg label="Tumour appearance at baseline" value={bb} onChange={setBb} /><Seg label="Tumour appearance at follow-up (a different contrast phase can change it)" value={bf} onChange={setBf} />
          </div>
          <div className="space-y-4" aria-live="polite">
            {res ? (
              <>
                <div className="glass p-4 sm:p-5"><div className="eyebrow">Truth in this simulated patient</div>
                  <p className="mt-1.5 text-[22px] font-medium">{NAME[res.truth.category]} <span className="num text-[15px] text-ink-dim">{res.truth.baseline_sum_mm.toFixed(0)} to {res.truth.followup_sum_mm.toFixed(0)} mm</span></p></div>
                {cd.flag ? <div role="status" className="rounded-xl border border-failure/50 bg-failure/[0.08] px-4 py-3 text-[14px] leading-relaxed text-ink"><strong className="font-medium text-failure">Response depends on calibration.</strong> The two thresholds give different categories in <span className="num">{pct(cd.probability_default_and_recalibrated_differ)}</span> of simulated draws. {cd.modal.default !== cd.modal.recal ? <>The default most often reports <span className="num">{cd.modal.default}</span>, the recalibrated threshold <span className="num">{cd.modal.recal}</span>.</> : <>Both most often report <span className="num">{cd.modal.default}</span>, but the call is not stable between the two.</>}</div>
                  : <div className="rounded-xl border border-white/10 px-4 py-3 text-[14px] text-ink-dim">The two thresholds agree here: they differ in only <span className="num text-ink">{pct(cd.probability_default_and_recalibrated_differ)}</span> of simulated draws.</div>}
                <div className="grid gap-4 sm:grid-cols-2"><Method title="Default threshold" m={res.methods.default} truth={res.truth.category} accent="#8A96A8" /><Method title="Recalibrated (research)" m={res.methods.recal} truth={res.truth.category} accent="#F5B54A" /></div>
                <div className="glass p-4 text-[12px] leading-relaxed text-ink-dim"><div className="eyebrow mb-1.5">Lesion matching (simulated positions)</div>
                  {res.pair.matching.matched.map(([i, j]) => <div key={i}>Baseline lesion {i + 1} <span className="num text-ink">{res.pair.baseline[i].diameter_mm.toFixed(0)} mm</span> matched to follow-up lesion {j + 1} <span className="num text-ink">{res.pair.followup[j].diameter_mm.toFixed(0)} mm</span></div>)}
                  <p className="mt-1.5">Matching uses lesion position and plausible size change, as it would on real scans after registration. Rule: {cd.rule}.</p></div>
              </>
            ) : <p className="text-ink-dim" role="status">Simulating…</p>}
            {err ? <p role="alert" className="text-failure">{err}</p> : null}
          </div>
        </div>
      ) : null}
      {model && model.available ? <p className="mt-8 max-w-3xl text-[12px] leading-relaxed text-ink-dim">The bias model was fitted on {model.n_lesions} real lesion measurements ({model.source}). Median error of detected lesions: default <span className="num">{model.median_error_mm_detected.default.toFixed(1)}</span> mm, recalibrated <span className="num">{model.median_error_mm_detected.recal.toFixed(1)}</span> mm. A simulation shows what the measurement error could do to a response call; it does not show what happened to any patient.</p> : null}
    </div>
  );
}
