import { useEffect, useRef, useState } from "react";
import { fileUrl, getJSON } from "../lib/api";
import { reducedMotion } from "../lib/motion";

const ORGAN_COLORS = [[94, 150, 230], [140, 120, 230], [230, 160, 90], [90, 200, 150], [210, 110, 150], [150, 190, 90], [90, 190, 210], [200, 190, 100], [170, 130, 200], [120, 160, 220]];

function makeLut(names) {
  const ids = Object.keys(names).map(Number).sort((a, b) => a - b), n = Math.max(...ids) + 1;
  const R = new Array(n).fill(0), G = new Array(n).fill(0), B = new Array(n).fill(0), A = new Array(n).fill(0), I = Array.from({ length: n }, (_, i) => i), labels = new Array(n).fill("");
  ids.forEach((id, k) => { const c = ORGAN_COLORS[k % ORGAN_COLORS.length]; R[id] = c[0]; G[id] = c[1]; B[id] = c[2]; A[id] = 255; labels[id] = names[id]; });
  return { R, G, B, A, I, labels };
}

const idx = (nv, name) => nv.volumes.findIndex((v) => v.name === name + ".nii.gz");
function ramp(nv, name, to, ms) {
  const i = idx(nv, name); if (i < 0) return; const from = nv.volumes[i].opacity;
  if (reducedMotion() || ms <= 0 || from === to) { nv.setOpacity(i, to); return; }
  const t0 = performance.now();
  const step = (t) => { const j = idx(nv, name); if (j < 0) return; const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); nv.setOpacity(j, from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

export const VIEWS = [["Axial", "sliceTypeAxial"], ["Coronal", "sliceTypeCoronal"], ["Sagittal", "sliceTypeSagittal"], ["Multi", "sliceTypeMultiplanar"], ["3D", "sliceTypeRender"]];
const FILL = 0.28;  // predicted tumour: semi-transparent fill plus a solid outline at full opacity

/** NiiVue viewer. Layers (only those the scan has): ct, organs, tumour (fill) + tumour_outline, tumour_recal (+outline), reference_outline (dashed, dataset annotation). */
export default function Viewer({ scanId, layers, mode, organsOn, predictedOn, referenceOn, view, focus, apiRef }) {
  const wrapRef = useRef(null), canvasRef = useRef(null), nvRef = useRef(null);
  const [ready, setReady] = useState(false), [loaded, setLoaded] = useState(""), [msg, setMsg] = useState("");

  useEffect(() => {
    let dead = false, ro = null;
    (async () => {
      try {
        const { Niivue } = await import("@niivue/niivue");
        if (dead) return;
        const nv = new Niivue({ backColor: [0.027, 0.035, 0.05, 1], crosshairColor: [0.37, 0.9, 0.82, 0.55], crosshairWidth: 0.5, show3Dcrosshair: false, isColorbar: false, isRuler: false, loglevel: "error", dragAndDropEnabled: false, multiplanarPadPixels: 6 });
        await nv.attachToCanvas(canvasRef.current);
        const solid = (r, g, b) => ({ R: [0, r], G: [0, g], B: [0, b], A: [0, 255], I: [0, 255] });
        nv.addColormap("accentmask", solid(94, 230, 208)); nv.addColormap("cautionmask", solid(245, 181, 74)); nv.addColormap("refmask", solid(232, 238, 245));
        nvRef.current = nv; window.__nv = nv; /* test hook */ if (apiRef) apiRef.current = { jump: (v) => jump(nv, v) };
        // keep the canvas matched to its box: the viewer column is sticky and full height, and its size settles after layout
        let raf = 0; ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { try { nv.resizeListener(); nv.drawScene(); } catch (e) { /* not drawn yet */ } }); });
        ro.observe(wrapRef.current); setReady(true);
      } catch (e) { setMsg("The 3D viewer could not start in this browser: " + e.message); }
    })();
    return () => { dead = true; if (ro) ro.disconnect(); };
  }, []);

  const has = (n) => layers.includes(n);
  useEffect(() => {
    const nv = nvRef.current; if (!ready || !nv) return; let dead = false;
    (async () => {
      setLoaded(""); setMsg("");
      try {
        const base = (n) => fileUrl(scanId, n), labels = await getJSON(base("labels.json"));
        while (nv.volumes.length) nv.removeVolume(nv.volumes[0]);
        const vol = (n, o) => ({ url: base(n + ".nii.gz"), name: n + ".nii.gz", opacity: 0, cal_min: 0, cal_max: 1, ...o });
        const vols = [{ url: base("ct.nii.gz"), name: "ct.nii.gz", colormap: "gray", cal_min: -150, cal_max: 250 }, vol("organs", { cal_max: 255, colormapLabel: makeLut(labels.organs) })];
        if (has("tumour")) vols.push(vol("tumour", { colormap: "accentmask" }));
        if (has("tumour_outline")) vols.push(vol("tumour_outline", { colormap: "accentmask" }));
        if (has("tumour_recal")) vols.push(vol("tumour_recal", { colormap: "cautionmask" }));
        if (has("tumour_recal_outline")) vols.push(vol("tumour_recal_outline", { colormap: "cautionmask" }));
        if (has("reference_outline")) vols.push(vol("reference_outline", { colormap: "refmask" }));
        await nv.loadVolumes(vols); if (dead) return;
        nv.setSliceType(nv.sliceTypeAxial); nv.resizeListener();
        if (focus) jump(nv, focus);
        setLoaded(scanId);
      } catch (e) { setMsg("Could not load the volume: " + e.message); }
    })();
    return () => { dead = true; };
  }, [ready, scanId, layers.join(",")]);

  // overlays fade / grow in over the scan; default <-> recalibrated cross-fades
  useEffect(() => {
    const nv = nvRef.current; if (!ready || loaded !== scanId) return;
    const d = predictedOn && mode === "default", r = predictedOn && mode === "recal";
    ramp(nv, "tumour", d ? FILL : 0, 600); ramp(nv, "tumour_outline", d ? 1 : 0, 600);
    ramp(nv, "tumour_recal", r ? FILL : 0, 600); ramp(nv, "tumour_recal_outline", r ? 1 : 0, 600);
    ramp(nv, "reference_outline", referenceOn ? 1 : 0, 400);
  }, [ready, loaded, scanId, predictedOn, referenceOn, mode]);
  useEffect(() => { const nv = nvRef.current; if (ready && loaded === scanId) ramp(nv, "organs", organsOn ? 0.32 : 0, 350); }, [ready, loaded, scanId, organsOn]);
  useEffect(() => { const nv = nvRef.current; if (ready && nv && nv[view]) nv.setSliceType(nv[view]); }, [ready, loaded, view]);

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden rounded-glass border border-white/[0.08] bg-[#07090d]">
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-label="CT viewer: scroll to change slice, drag to move the cursor" role="img" />
      {!loaded && !msg ? <div className="absolute inset-0 grid place-items-center text-[13px] text-ink-dim" role="status">Loading the scan…</div> : null}
      {msg ? <div className="absolute inset-x-0 bottom-3 px-4 text-center text-[13px] text-failure" role="alert">{msg}</div> : null}
    </div>
  );
}

function jump(nv, vox) {
  if (!nv || !nv.volumes.length || !vox) return;
  try { nv.scene.crosshairPos = nv.vox2frac(vox, 0); nv.updateGLVolume(); } catch (e) { /* cursor stays where it is */ }
}
