import { motion } from "framer-motion";
import { EASE } from "../lib/motion";

/** Fade + 8px rise when scrolled into view (disabled under prefers-reduced-motion via MotionConfig). */
export default function Reveal({ children, delay = 0, className = "" }) {
  return <motion.div className={className} initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-8% 0px" }} transition={{ duration: 0.4, ease: EASE, delay }}>{children}</motion.div>;
}
