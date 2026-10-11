import { useEffect, useState } from "react";
import Reveal from "../components/Reveal";
import { getHealth } from "../lib/api";

const Card = ({ title, children }) => <Reveal className="glass p-5 sm:p-6"><h2 className="eyebrow">{title}</h2><div className="mt-3 space-y-3 text-[14px] leading-relaxed text-ink-dim">{children}</div></Reveal>;
const L = ({ href, children }) => <a className="focus-ring rounded text-accent underline-offset-2 hover:underline" href={href} target="_blank" rel="noopener noreferrer">{children}</a>;

export default function About() {
  const [h, setH] = useState(null);
  useEffect(() => { getHealth().then(setH).catch(() => {}); }, []);
  return (
    <div className="mx-auto max-w-[1000px] px-4 pt-12 sm:px-6 sm:pt-20">
      <div className="eyebrow">About</div>
      <h1 className="mt-3 text-[34px] font-medium leading-[1.08] tracking-tight sm:text-[48px]">A research instrument, not a medical device.</h1>
      <div className="mt-10 grid gap-4 md:grid-cols-2">
        <Card title="Research use only"><p><strong className="font-medium text-ink">Not for clinical decisions.</strong> Do not upload patient data. Everything here describes how a model behaves on public research scans; it says nothing about any patient.</p></Card>
        <Card title="Data and attribution">
          <p>Sample scans are from <strong className="font-medium text-ink">HCC-TACE-Seg</strong> on The Cancer Imaging Archive, licensed CC BY 4.0. They were resampled for viewing; the segmentations shown are the model's output, not the dataset's annotations.</p>
          <ul className="space-y-2 text-[13px]">
            <li>Moawad, A. W. et al. (2021). Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg) [Data set]. TCIA. <L href="https://doi.org/10.7937/TCIA.5FNA-0924">doi:10.7937/TCIA.5FNA-0924</L></li>
            <li>Moawad, A. W. et al. (2023). Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation. <em>Scientific Data</em> 10, 33. <L href="https://doi.org/10.1038/s41597-023-01928-3">doi:10.1038/s41597-023-01928-3</L></li>
            <li>Clark, K. et al. (2013). The Cancer Imaging Archive (TCIA). <em>J Digit Imaging</em> 26(6), 1045–1057. <L href="https://doi.org/10.1007/s10278-013-9622-7">doi:10.1007/s10278-013-9622-7</L></li>
          </ul>
        </Card>
        <Card title="How NVIDIA models are used"><p>The segmentation is <strong className="font-medium text-ink">NVIDIA NV-Segment-CT (VISTA-3D)</strong>, used frozen under the NVIDIA Open Model License. The research opens this model up: probes, erasure, patching and steering on its internal activations. During development the agent's language model ran on NVIDIA's hosted API.</p></Card>
        <Card title="How Nebius is used"><p>The agent that gathers the tool results and writes the report talks to an OpenAI-compatible endpoint set entirely by configuration. In the submitted build that endpoint is <strong className="font-medium text-ink">Nebius Token Factory</strong>, and the app can be hosted on Nebius AI Cloud.</p>
          <p className="rounded-lg border border-white/10 p-3 text-[12px]">This server right now: {h ? (h.language_model ? <>language model <span className="num text-ink">{h.provider || "configured"}</span> · <span className="num text-ink">{h.model || ""}</span></> : <>no language model configured, reports use the deterministic template</>) : "checking…"}</p></Card>
        <Card title="Code and findings"><p>Source, pre-registrations, deviations and the full research summary: <L href="https://github.com/Mokhan02/Glass-Box-CT">github.com/Mokhan02/Glass-Box-CT</L>. MIT licensed. The NVIDIA model weights keep their own licence.</p></Card>
        <Card title="Known limits"><p>This model is known to miss some liver tumours brighter than the surrounding liver (research finding, n=89). A small or absent tumour prediction does not rule out a tumour. The quality flag is a research signal fitted on a small held-out set.</p></Card>
      </div>
    </div>
  );
}
