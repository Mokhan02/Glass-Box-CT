import { useEffect, useState } from "react";

export const EASE = [0.16, 1, 0.3, 1];
export const reducedMotion = () => typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** page / route transition: fade + 8px rise, 250 ms ease-out */
export const pageVariants = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0, transition: { duration: 0.25, ease: EASE } }, exit: { opacity: 0, transition: { duration: 0.12 } } };

export function useReduced() {
  const [r, setR] = useState(reducedMotion());
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)"); const f = () => setR(m.matches);
    m.addEventListener("change", f); return () => m.removeEventListener("change", f);
  }, []);
  return r;
}
