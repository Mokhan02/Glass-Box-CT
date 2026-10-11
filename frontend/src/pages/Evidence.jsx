import { Link } from "react-router-dom";
import Reveal from "../components/Reveal";
import RangeBars from "../components/RangeBars";
import CausalChain from "../components/CausalChain";
import f from "../data/findings.json";
import { minus } from "../lib/format";

const A = "#5EE6D0", C = "#F5B54A", F = "#FF7A6B", D = "#8A96A8";

function Section({ n, title, lede, children }) {
  return (
    <section className="grid gap-8 border-t border-white/[0.07] py-14 md:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] md:gap-14 md:py-20" aria-labelledby={`s${n}`}>
      <Reveal>
        <div className="eyebrow">Finding {n}</div>
        <h2 id={`s${n}`} className="mt-2 text-[28px] font-medium leading-tight tracking-tight sm:text-[34px]">{title}</h2>
        <p className="mt-4 text-[15px] leading-relaxed text-ink-dim">{lede}</p>
      </Reveal>
      <Reveal delay={0.08} className="glass self-start p-5 sm:p-6">{children}</Reveal>
    </section>
  );
}
const Tag = ({ kind }) => <span className="num ml-2 rounded-full border px-2 py-0.5 align-middle text-[10px] uppercase tracking-wider" style={{ color: kind === "expl" ? C : A, borderColor: kind === "expl" ? "rgba(245,181,74,.4)" : "rgba(94,230,208,.4)" }}>{kind === "expl" ? "exploratory" : "pre-registered"}</span>;

export default function Evidence() {
  const b = f.bias, p = f.probe, br = f.brightness, e = f.erasure, pa = f.patching;
  return (
    <div className="mx-auto max-w-[1200px] px-4 pt-12 sm:px-6 sm:pt-20">
      <header className="max-w-3xl">
        <div className="eyebrow">Evidence</div>
        <h1 className="mt-3 text-[34px] font-medium leading-[1.08] tracking-tight sm:text-[52px]">The model sees the tumour, then loses it.</h1>
        <p className="mt-5 text-[16px] leading-relaxed text-ink-dim">Everything below was measured on held-out research CT, and every number comes from the project's research summary. These are statements about the model, not about patients.</p>
      </header>

      <Section n={1} title="It measures tumours too small." lede="The size a tumour is reported at decides how a change over time is read. The model under-measures, and almost never over-measures.">
        <div className="eyebrow">{b.sld.label}</div>
        <div className="mt-2 flex items-baseline gap-3"><span className="num text-[56px] leading-none text-failure">{minus(b.sld.value)}<span className="unit text-[28px]">{b.sld.unit}</span></span></div>
        <div className="num mt-1 text-[13px] text-ink-dim">95% CI [{minus(b.sld.ci[0])}, {minus(b.sld.ci[1])}] · {b.sld.cohort}</div>
        <div className="mt-6 grid grid-cols-2 gap-5 border-t border-white/[0.07] pt-5">
          <div><div className="num text-[26px]">{minus(b.detected_mm.value)}<span className="unit text-[14px]"> {b.detected_mm.unit}</span></div><div className="mt-1 text-[12px] leading-snug text-ink-dim">{b.detected_mm.label} ({b.detected_mm.note})</div></div>
          <div><div className="num text-[26px]">{b.missed.value}</div><div className="mt-1 text-[12px] leading-snug text-ink-dim">{b.missed.label}</div></div>
        </div>
        <div className="mt-6 rounded-xl border border-white/10 p-4">
          <div className="eyebrow">One fix, and its cost</div>
          <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">A single lower threshold moves the median error from <span className="num text-ink">{minus(b.orig.sld)}%</span> to <span className="num text-ink">{minus(b.recal.sld)}%</span> and the overlap with the reference from <span className="num text-ink">{b.orig.dice}</span> to <span className="num text-ink">{b.recal.dice}</span> (Dice) — but it overestimates by ≥20% in <span className="num text-caution">{b.recal.over}%</span> of cases.</p>
        </div>
      </Section>

      <Section n={2} title="It knows more than it says." lede="A simple read-out of the network's middle layers finds the tissue the final answer misses. The same read-out on an untrained network, or on raw brightness, finds almost none of it. The information is there; it is lost late.">
        <div className="eyebrow mb-3">Missed tumour tissue found by a read-out of each layer (fraction)<Tag kind="pre" /></div>
        <RangeBars domain={[0, 1]} zero={false} ariaLabel="Detection of missed tumour tissue: trained network versus untrained network versus raw brightness, at two layers"
          rows={[{ label: "Trained, mid layer", ...p.mid.trained, color: A }, { label: "Untrained network", ...p.mid.random, color: D }, { label: "Raw brightness", ...p.mid.hu, color: D },
            { label: "Trained, deeper layer", ...p.bottleneck.trained, color: A }, { label: "Untrained network", ...p.bottleneck.random, color: D }, { label: "Raw brightness", ...p.bottleneck.hu, color: D }]} />
        <p className="mt-5 text-[13px] leading-relaxed text-ink-dim">The final scores of missed tissue sit about <span className="num text-ink">{p.logit.value}</span> logits below the cut-off: close enough that a threshold can recover some of it.</p>
      </Section>

      <Section n={3} title="Brighter than the liver, more often missed." lede="Edit one tumour's brightness inside its outline and change nothing else. The model's answer follows the brightness.">
        <div className="eyebrow mb-3">Change in how much of the tumour is found<Tag kind="pre" /></div>
        <RangeBars domain={[-0.6, 0.6]} ariaLabel="Change in tumour coverage when tumours are edited from dark to bright, and from bright to dark"
          rows={[{ label: `Dark made bright (n=${br.armA.n})`, ...br.armA, color: F }, { label: `…minus a control edit`, ...br.armA_minus_control, color: F }, { label: `Bright made dark (n=${br.armB.n})`, ...br.armB, color: A }]} />
        <p className="mt-4 text-[13px] leading-relaxed text-ink-dim">Strong in the dark-to-bright direction (<span className="num text-ink">{br.armA.hits}</span> tumours move as predicted). The reverse passes narrowly: the lower bound of its interval is <span className="num text-caution">{br.armB.ci[0]}</span>.</p>
        <p className="mt-3 text-[13px] leading-relaxed text-ink-dim">In the real scans, <span className="num text-ink">{br.bright_det.value}%</span> of bright-tumour lesions are found against <span className="num text-ink">{br.dark_det.value}%</span> of dark-tumour lesions.<Tag kind="expl" /></p>
      </Section>

      <Section n={4} title="One direction decides it." lede="Remove a single direction in the middle of the network, the one that separates tumour from liver, and the model loses tumours it found before. Remove a random direction of the same size and nothing happens.">
        <div className="eyebrow mb-3">Change in overlap on tumours the model used to find<Tag kind="pre" /></div>
        <RangeBars domain={[-0.5, 0.1]} ariaLabel="Change in lesion Dice after erasing the tumour direction, a random direction, or a brightness-only direction"
          rows={[{ label: "Tumour direction removed", ...e.leace, color: F }, { label: "Random direction", value: e.random.value, color: D }, { label: "Brightness-only direction", value: e.hu.value, color: D }]} />
        <p className="mt-4 text-[13px] leading-relaxed text-ink-dim">Afterwards only <span className="num text-ink">{e.leace.still}%</span> of those tumours are still found, while the liver outline does not change. In the decoder, <span className="num text-ink">{e.channels.n}</span> channels carry <span className="num text-ink">{e.channels.value}</span> of the gap between found and missed tissue.</p>
      </Section>

      <Section n={5} title="It happens inside the tumour, early." lede="Run the bright scan, but paste in the network's internal state from the dark version of the same tumour. Where the pasted state restores the detection tells you where the problem lives.">
        <div className="eyebrow mb-3">Share of the lost detection restored<Tag kind="pre" /></div>
        <RangeBars domain={[0, 1]} zero={false} ariaLabel="Fraction of the lost detection restored by pasting the dark version's state, by layer and region"
          rows={[{ label: "Tumour, mid layer", ...pa.mid, color: A }, { label: "Tumour, deeper layer", ...pa.bottleneck, color: A }, { label: "Surrounding shell only", value: pa.shell.value, color: D }, { label: "Everywhere else", value: pa.far.value, color: D }]} />
        <p className="mt-4 text-[13px] leading-relaxed text-ink-dim">Making the tumour brighter lowers its tumour feature in <span className="num text-ink">{pa.bridge.hits}</span> tumours, and the bigger the drop, the more detection is lost (rank correlation <span className="num text-ink">{pa.bridge.rho}</span>).<Tag kind="expl" /></p>
      </Section>

      <section className="border-t border-white/[0.07] py-14 md:py-20" aria-labelledby="chain">
        <Reveal><div className="eyebrow">The chain</div><h2 id="chain" className="mt-2 text-[28px] font-medium tracking-tight sm:text-[34px]">From brightness to a missed tumour.</h2></Reveal>
        <div className="mt-8"><CausalChain /></div>
        <Reveal className="mt-10"><p className="max-w-2xl text-[13px] leading-relaxed text-ink-dim">Limits: this is one model on one disease in one public dataset, and edits are synthetic. A steering test that adds the tumour direction back behaved like a plain threshold shift, so it is not a better detector. <Link to="/" className="focus-ring rounded text-accent hover:underline">Try the scans</Link>.</p></Reveal>
      </section>
    </div>
  );
}
