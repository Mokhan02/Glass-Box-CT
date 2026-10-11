import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { fileUrl } from "../lib/api";
import Chip from "./Chip";

/** The five scans as compact cards (thumbnail, chip, one line), to move between scans without going back to the landing page. */
export default function ScanSwitcher({ scans, current }) {
  const ref = useRef(null);
  useEffect(() => { const el = ref.current && ref.current.querySelector("[aria-current=page]"); if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest", inline: "center" }); }, [current, scans.length]);
  if (!scans.length) return null;
  return (
    <nav aria-label="Scans">
      <ul ref={ref} className="-mx-1 flex snap-x gap-2 overflow-x-auto px-1 pb-2">
        {scans.map((s) => {
          const on = s.id === current;
          return (
            <li key={s.id} className="w-[148px] shrink-0 snap-start">
              <Link to={`/scan/${s.id}`} aria-current={on ? "page" : undefined} className={`focus-ring block overflow-hidden rounded-xl border transition-colors duration-200 ${on ? "border-accent/70 bg-accent/[0.06]" : "border-white/10 hover:border-white/25"}`}>
                <div className="relative aspect-[16/9] bg-black"><img src={fileUrl(s.id, "thumb.png")} alt="" loading="lazy" className="h-full w-full object-cover opacity-90" /><div className="absolute left-1.5 top-1.5 scale-90 origin-top-left"><Chip kind={s.chip} /></div></div>
                <div className="px-2 py-1.5"><div className="eyebrow !text-[10px]">Scan {s.id.replace("demo_", "")}</div><p className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug text-ink">{s.oneliner}</p></div>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
