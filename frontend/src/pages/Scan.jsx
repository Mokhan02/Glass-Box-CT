import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Viewer, { VIEWS } from "../components/Viewer";
import AgentPanel from "../components/AgentPanel";
import { getReport, getSummary, listScans } from "../lib/api";

const B = "btn-pill !border-transparent !px-2.5 !py-1 !text-[12px]";

function Pill({ view, setView, organsOn, setOrgansOn, predictedOn, setPredictedOn, referenceOn, setReferenceOn, hasReference, mode }) {
  return (
    <div className="glass absolute inset-x-3 top-3 z-10 mx-auto flex w-fit max-w-[calc(100%-1.5rem)] flex-wrap items-center justify-center gap-0.5 rounded-full p-1" role="toolbar" aria-label="Viewer controls">
      {VIEWS.map(([label, key]) => <button key={key} className={B} aria-pressed={view === key} onClick={() => setView(key)}>{label}</button>)}
      <span className="mx-1 h-4 w-px bg-white/15" aria-hidden="true" />
      <button className={B} aria-pressed={organsOn} onClick={() => setOrgansOn(!organsOn)}>Organs</button>
      <button className={B} aria-pressed={predictedOn} onClick={() => setPredictedOn(!predictedOn)}>{mode === "recal" ? "Predicted tumour (recalibrated)" : "Predicted tumour"}</button>
      <button className={`${B} disabled:opacity-40`} aria-pressed={referenceOn} disabled={!hasReference} title={hasReference ? "The dataset's own tumour annotation, dashed" : "The reference annotation is not installed for this scan"} onClick={() => setReferenceOn(!referenceOn)}>Reference tumour (dataset)</button>
    </div>
  );
}

export default function ScanPage() {
  const { id } = useParams();
  const [meta, setMeta] = useState(null), [all, setAll] = useState([]), [summary, setSummary] = useState(null), [report, setReport] = useState(null), [error, setError] = useState("");
  const [mode, setMode] = useState("default"), [view, setView] = useState("sliceTypeAxial"), [organsOn, setOrgansOn] = useState(false), [predictedOn, setPredictedOn] = useState(true), [referenceOn, setReferenceOn] = useState(false);
  const viewerApi = useRef(null);

  useEffect(() => {
    let dead = false; setSummary(null); setReport(null); setError(""); setMode("default"); setView("sliceTypeAxial"); setMeta(null);
    listScans().then((d) => { if (dead) return; setAll(d.scans || []); setMeta((d.scans || []).find((s) => s.id === id) || { id }); }).catch(() => {});
    (async () => {
      try {
        const s = await getSummary(id); if (dead) return; setSummary(s);
        const r = await getReport(id); if (dead) return; setReport(r);
      } catch (e) { if (!dead) setError(e.message); }
    })();
    return () => { dead = true; };
  }, [id]);

  const layers = (summary && summary.layers) || [], hasReference = layers.includes("reference_outline");
  useEffect(() => { setReferenceOn(!!(meta && meta.reference_default && hasReference)); }, [meta, hasReference]);  // demos where the model misses the tumour show the dataset's reference by default
  const focus = summary && summary.lesions && summary.lesions.lesions && summary.lesions.lesions[0] ? summary.lesions.lesions[0].voxel_in_file : null;
  return (
    <div className="mx-auto max-w-[1700px] px-3 pt-3 sm:px-5">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,60fr)_minmax(0,40fr)]">
        <div className="relative h-[64vh] min-h-[380px] lg:sticky lg:top-[68px] lg:h-[calc(100vh-84px)] lg:self-start">
          {summary ? <Viewer scanId={id} layers={layers} mode={mode} organsOn={organsOn} predictedOn={predictedOn} referenceOn={referenceOn} view={view} focus={focus} apiRef={viewerApi} /> : <div className="grid h-full place-items-center rounded-glass border border-white/[0.08] text-[13px] text-ink-dim" role="status">Loading…</div>}
          <Pill view={view} setView={setView} organsOn={organsOn} setOrgansOn={setOrgansOn} predictedOn={predictedOn} setPredictedOn={setPredictedOn} referenceOn={referenceOn} setReferenceOn={setReferenceOn} hasReference={hasReference} mode={mode} />
          <ul className="pointer-events-none absolute bottom-3 left-4 right-4 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-dim">
            <li className="flex items-center gap-1.5"><span aria-hidden="true" className="inline-block h-2.5 w-4 rounded-sm border-2" style={{ borderColor: mode === "recal" ? "#F5B54A" : "#5EE6D0", background: mode === "recal" ? "rgba(245,181,74,.3)" : "rgba(94,230,208,.3)" }} />predicted tumour{mode === "recal" ? " (recalibrated)" : ""}</li>
            {hasReference ? <li className="flex items-center gap-1.5"><span aria-hidden="true" className="inline-block h-2.5 w-4 rounded-sm border-2 border-dashed border-ink" />reference tumour (dataset annotation)</li> : null}
            <li>scroll to change slice</li>
          </ul>
        </div>
        <aside className="glass p-5 lg:min-h-[calc(100vh-84px)]" aria-label="Agent and report">
          <AgentPanel meta={meta} all={all} summary={summary} report={report} error={error} mode={mode} setMode={setMode} onJump={(v) => viewerApi.current && viewerApi.current.jump(v)} />
        </aside>
      </div>
    </div>
  );
}
