const STYLES = {
  trusted: { color: "#5EE6D0", bg: "rgba(94,230,208,0.10)", border: "rgba(94,230,208,0.35)", label: "Trusted" },
  review: { color: "#F5B54A", bg: "rgba(245,181,74,0.10)", border: "rgba(245,181,74,0.35)", label: "Review" },
  unreliable: { color: "#FF7A6B", bg: "rgba(255,122,107,0.10)", border: "rgba(255,122,107,0.35)", label: "Unreliable" },
  good: { color: "#5EE6D0", bg: "rgba(94,230,208,0.10)", border: "rgba(94,230,208,0.35)", label: "Good" },
  flagged: { color: "#F5B54A", bg: "rgba(245,181,74,0.10)", border: "rgba(245,181,74,0.35)", label: "Flagged" },
  missed: { color: "#FF7A6B", bg: "rgba(255,122,107,0.10)", border: "rgba(255,122,107,0.35)", label: "Missed" },
};

/** Status chip: colour is always paired with a text label. */
export default function Chip({ kind, children }) {
  const s = STYLES[String(kind || "").toLowerCase()] || STYLES.flagged;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[12px] font-medium" style={{ color: s.color, background: s.bg, border: `1px solid ${s.border}` }}>
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full" style={{ background: s.color }} />
      {children || s.label}
    </span>
  );
}
