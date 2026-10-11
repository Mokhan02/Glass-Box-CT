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

function ramp(nv, idx, from, to, ms) {
  if (!nv || !nv.volumes[idx]) return;
  if (reducedMotion() || ms <= 0) { nv.setOpacity(idx, to); return; }
  const t0 = performance.now();
  const step = (t) => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); if (nv.volumes[idx]) nv.setOpacity(idx, from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

export const VIEWS = [["Axial", "sliceTypeAxial"], ["Coronal", "sliceTypeCoronal"], ["Sagittal", "sliceTypeSagittal"], ["Multi", "sliceTypeMultiplanar"], ["3D", "sliceTypeRender"]];
const TUMOUR_OPACITY = 0.78, ORGAN_OPACITY = 0.32;

/** NiiVue viewer. Volumes: 0 CT, 1 organs, 2 predicted tumour (default threshold), 3 predicted tumour (recalibrated), if the scan has it. */
export default function Viewer({ scanId, hasRecal, mode, organsOn, tumourOn, view, focus, apiRef }) {
  const canvasRef = useRef(null), nvRef = useRef(null);
  const [ready, setReady] = useState(false), [loaded, setLoaded] = useState(""), [msg, setMsg] = useState("");
  const shown = useRef({ default: 0, recal: 0 });

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const { Niivue } = await import("@niivue/niivue");
        if (dead) return;
        const nv = new Niivue({ backColor: [0.027, 0.035, 0.05, 1], crosshairColor: [0.37, 0.9, 0.82, 0.9], crosshairWidth: 0.6, show3Dcrosshair: false, isColorbar: false, isRuler: false, loglevel: "error", dragAndDropEnabled: false });
        await nv.attachToCanvas(canvasRef.current);
        nv.addColormap("accentmask", { R: [0, 94], G: [0, 230], B: [0, 208], A: [0, 255], I: [0, 255] });
        nv.addColormap("cautionmask", { R: [0, 245], G: [0, 181], B: [0, 74], A: [0, 255], I: [0, 255] });
        nvRef.current = nv; window.__nv = nv; /* test hook */ if (apiRef) apiRef.current = { jump: (v) => jump(nv, v) };
        setReady(true);
      } catch (e) { setMsg("The 3D viewer could not start in this browser: " + e.message); }
    })();
    return () => { dead = true; };
  }, []);

  useEffect(() => {
    const nv = nvRef.current; if (!ready || !nv) return; let dead = false;
    (async () => {
      setLoaded(""); setMsg("");
      try {
        const base = (n) => fileUrl(scanId, n), labels = await getJSON(base("labels.json"));
        while (nv.volumes.length) nv.removeVolume(nv.volumes[0]);
        const vols = [{ url: base("ct.nii.gz"), name: "ct.nii.gz", colormap: "gray", cal_min: -150, cal_max: 250 },
          { url: base("organs.nii.gz"), name: "organs.nii.gz", cal_min: 0, cal_max: 255, colormapLabel: makeLut(labels.organs), opacity: 0 },
          { url: base("tumour.nii.gz"), name: "tumour.nii.gz", colormap: "accentmask", cal_min: 0, cal_max: 1, opacity: 0 }];
        if (hasRecal) vols.push({ url: base("tumour_recal.nii.gz"), name: "tumour_recal.nii.gz", colormap: "cautionmask", cal_min: 0, cal_max: 1, opacity: 0 });
        await nv.loadVolumes(vols); if (dead) return;
        nv.setSliceType(nv.sliceTypeAxial); shown.current = { default: 0, recal: 0 };
        if (focus) jump(nv, focus);
        setLoaded(scanId);
      } catch (e) { setMsg("Could not load the volume: " + e.message); }
    })();
    return () => { dead = true; };
  }, [ready, scanId, hasRecal]);

  // overlays: tumour fades / grows in over the scan; switching default <-> recalibrated cross-fades
  useEffect(() => {
    const nv = nvRef.current; if (!ready || loaded !== scanId) return;
    const wantD = tumourOn && mode === "default" ? TUMOUR_OPACITY : 0, wantR = tumourOn && mode === "recal" && hasRecal ? TUMOUR_OPACITY : 0;
    ramp(nv, 2, shown.current.default, wantD, 600); shown.current.default = wantD;
    if (hasRecal) { ramp(nv, 3, shown.current.recal, wantR, 600); shown.current.recal = wantR; }
  }, [ready, loaded, scanId, tumourOn, mode, hasRecal]);
  useEffect(() => { const nv = nvRef.current; if (ready && loaded === scanId) ramp(nv, 1, nv.volumes[1] ? nv.volumes[1].opacity : 0, organsOn ? ORGAN_OPACITY : 0, 350); }, [ready, loaded, scanId, organsOn]);
  useEffect(() => { const nv = nvRef.current; if (ready && nv && nv[view]) nv.setSliceType(nv[view]); }, [ready, loaded, view]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-glass border border-white/[0.08] bg-black">
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
