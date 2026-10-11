import { useState } from "react";
import Chip from "./Chip";

const isFragment = (l) => l.longest_axial_diameter.value < 3 || l.volume.value < 0.1;

/** Lesions sorted by size; fragments (< 3 mm or < 0.1 cm3) are collapsed into a single row. */
export default function LesionsTable({ lesions, trust, onJump }) {
  const tl = Object.fromEntries(((trust && trust.lesions) || []).map((x) => [x.id, x]));
  const [open, setOpen] = useState(false);
  const sorted = [...lesions].sort((a, b) => b.longest_axial_diameter.value - a.longest_axial_diameter.value), main = sorted.filter((l) => !isFragment(l)), frags = sorted.filter(isFragment);
  const row = (l, dim) => (
    <tr key={l.id} className={`border-t border-white/[0.06] ${dim ? "text-ink-dim" : ""}`}>
      <td className="num py-2 pr-3">{l.longest_axial_diameter.value}<span className="unit"> mm</span></td>
      <td className="num py-2 pr-3">{l.volume.value}<span className="unit"> cm3</span></td>
      <td className="py-2 pr-3 text-ink-dim">{l.recist.replace("non-target (more than 2 larger lesions in this organ)", "non-target").replace("non-target (below 10 mm)", "non-target (< 10 mm)")}</td>
      <td className="py-2 pr-3" title={tl[l.id] ? tl[l.id].reasons.map((r) => r.text).join(" ") : ""}>{tl[l.id] ? <Chip kind={tl[l.id].level} /> : null}</td>
      <td className="py-2 text-right"><button className="btn-pill !py-0.5 !text-[12px]" onClick={() => onJump(l.voxel_in_file)} aria-label={`Show lesion of ${l.longest_axial_diameter.value} millimetres in the viewer`}>show</button></td>
    </tr>
  );
  if (!lesions.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px]">
        <caption className="sr-only">Predicted lesions, largest first</caption>
        <thead><tr className="text-left"><th scope="col" className="eyebrow whitespace-nowrap pb-1.5 pr-3 font-normal">Longest diameter</th><th scope="col" className="eyebrow whitespace-nowrap pb-1.5 pr-3 font-normal">Volume</th><th scope="col" className="eyebrow whitespace-nowrap pb-1.5 pr-3 font-normal">RECIST 1.1</th><th scope="col" className="eyebrow whitespace-nowrap pb-1.5 pr-3 font-normal">Verdict</th><th /></tr></thead>
        <tbody>
          {main.map((l) => row(l))}
          {frags.length ? (
            <>
              <tr className="border-t border-white/[0.06]"><td colSpan={4} className="py-2 text-ink-dim"><button className="focus-ring rounded text-left hover:text-ink" aria-expanded={open} onClick={() => setOpen(!open)}><span aria-hidden="true">{open ? "▾" : "▸"}</span> {frags.length} fragment{frags.length > 1 ? "s" : ""} <span className="text-[12px]">(under 3 mm or 0.1 cm3)</span></button></td><td /></tr>
              {open ? frags.map((l) => row(l, true)) : null}
            </>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}
