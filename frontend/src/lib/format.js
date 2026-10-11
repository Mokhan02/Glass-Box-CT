/** Display helpers. The minus sign is typographic; numbers are formatted but never altered. */
export const minus = (s) => String(s).replace(/^-/, "−");
export const fmt = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "—");
export const sign = (s) => minus(String(s));
