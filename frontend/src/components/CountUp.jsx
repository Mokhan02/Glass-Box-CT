import { useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";
import { reducedMotion } from "../lib/motion";
import { minus } from "../lib/format";

/** Mono number that counts up to `value` in ~600 ms on mount / when the value changes. Reduced motion: shows the final value. */
export default function CountUp({ value, decimals = 1, duration = 0.6, className = "" }) {
  const [shown, setShown] = useState(reducedMotion() ? value : 0);
  const prev = useRef(reducedMotion() ? value : 0);
  useEffect(() => {
    if (reducedMotion() || !Number.isFinite(value)) { setShown(value); prev.current = value; return; }
    const c = animate(prev.current, value, { duration, ease: [0.16, 1, 0.3, 1], onUpdate: (v) => setShown(v), onComplete: () => { prev.current = value; } });
    return () => c.stop();
  }, [value, duration]);
  return <span className={`num ${className}`}>{minus(Number.isFinite(shown) ? shown.toFixed(decimals) : "—")}</span>;
}
