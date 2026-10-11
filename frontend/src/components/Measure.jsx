import CountUp from "./CountUp";

/** Large mono measurement: tabular number, unit in the secondary colour. */
export default function Measure({ label, value, unit, decimals = 1, hint, size = "text-[34px]" }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow">{label}</div>
      <div className={`mt-1 leading-none ${size}`} aria-label={`${label}: ${value} ${unit}`}>
        <CountUp value={value} decimals={decimals} /> <span className="unit text-[15px]">{unit}</span>
      </div>
      {hint ? <div className="mt-1.5 text-[12px] text-ink-dim">{hint}</div> : null}
    </div>
  );
}
