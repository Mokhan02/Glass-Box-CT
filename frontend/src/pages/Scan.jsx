import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Viewer, { VIEWS } from "../components/Viewer";
import AgentPanel from "../components/AgentPanel";
import { getReport, getSummary, listScans } from "../lib/api";

function Pill({ view, setView, organsOn, setOrgansOn, tumourOn, setTumourOn, mode }) {
  return (
    <div className="glass absolute inset-x-3 top-3 z-10 mx-auto flex w-fit max-w-[calc(100%-1.5rem)] flex-wrap items-center justify-center gap-0.5 rounded-full p-1" role="toolbar" aria-label="Viewer controls">
      {VIEWS.map(([label, key]) => <button key={key} className="btn-pill !border-transparent !px-2.5 !py-1 !text-[12px]" aria-pressed={view === key} onClick={() => setView(key)}>{label}</button>)}
      <span className="mx-1 h-4 w-px bg-white/15" aria-hidden="true" />
      <button className="btn-pill !border-transparent !px-2.5 !py-1 !text-[12px]" aria-pressed={organsOn} onClick={() => setOrgansOn(!organsOn)}>Organs</button>
      <button className="btn-pill !border-transparent !px-2.5 !py-1 !text-[12px]" aria-pressed={tumourOn} onClick={() => setTumourOn(!tumourOn)}>{mode === "recal" ? "Tumour (recalibrated)" : "Tumour"}</button>
    </div>
  );
}

export default function ScanPage() {
  const { id } = useParams();
  const [meta, setMeta] = useState(null), [summary, setSummary] = useState(null), [report, setReport] = useState(null), [error, setError] = useState("");
  const [mode, setMode] = useState("default"), [view, setView] = useState("sliceTypeAxial"), [organsOn, setOrgansOn] = useState(false), [tumourOn, setTumourOn] = useState(true);
  const viewerApi = useRef(null);

  useEffect(() => {
    let dead = false; setSummary(null); setReport(null); setError(""); setMode("default"); setView("sliceTypeAxial");
    listScans().then((d) => !dead && setMeta((d.scans || []).find((s) => s.id === id) || { id })).catch(() => {});
    (async () => {
      try {
        const s = await getSummary(id); if (dead) return; setSummary(s);
        const r = await getReport(id); if (dead) return; setReport(r);
      } catch (e) { if (!dead) setError(e.message); }
    })();
    return () => { dead = true; };
  }, [id]);

  const focus = summary && summary.lesions && summary.lesions.lesions && summary.lesions.lesions[0] ? summary.lesions.lesions[0].voxel_in_file : null;
  return (
    <div className="mx-auto max-w-[1700px] px-3 pt-3 sm:px-5">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <Link to="/" className="focus-ring rounded text-[13px] text-ink-dim hover:text-ink">← All scans</Link>
        <h1 className="text-[15px] text-ink">{meta && (meta.oneliner || meta.title)}</h1>
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,60fr)_minmax(0,40fr)]">
        <div className="relative h-[62vh] min-h-[360px] lg:sticky lg:top-[68px] lg:h-[calc(100vh-130px)]">
          {summary ? <Viewer scanId={id} hasRecal={!!summary.recalibrated} mode={mode} organsOn={organsOn} tumourOn={tumourOn} view={view} focus={focus} apiRef={viewerApi} /> : <div className="grid h-full place-items-center rounded-glass border border-white/[0.08] text-[13px] text-ink-dim" role="status">Loading…</div>}
          <Pill view={view} setView={setView} organsOn={organsOn} setOrgansOn={setOrgansOn} tumourOn={tumourOn} setTumourOn={setTumourOn} mode={mode} />
          <p className="pointer-events-none absolute bottom-3 left-4 right-4 text-[11px] text-ink-dim"><span className="text-accent">■</span> predicted tumour{summary && summary.recalibrated ? <> · <span className="text-caution">■</span> recalibrated</> : null} · scroll to change slice</p>
        </div>
        <aside className="glass p-5 lg:max-h-[calc(100vh-130px)] lg:overflow-y-auto" aria-label="Agent and report">
          {meta && meta.blurb ? <p className="mb-4 text-[12px] leading-relaxed text-ink-dim">{meta.blurb}</p> : null}
          <AgentPanel summary={summary} report={report} error={error} mode={mode} setMode={setMode} onJump={(v) => viewerApi.current && viewerApi.current.jump(v)} />
        </aside>
      </div>
    </div>
  );
}
