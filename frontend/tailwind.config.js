/** Design tokens, defined once. See frontend/src/index.css for the CSS variables the glass panels use. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#07090D",
        ink: { DEFAULT: "#E8EEF5", dim: "#8A96A8" },
        accent: "#5EE6D0",
        caution: "#F5B54A",
        failure: "#FF7A6B",
      },
      fontFamily: {
        sans: ['"Geist Variable"', "Inter", "system-ui", "sans-serif"],
        mono: ['"Geist Mono Variable"', '"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: { glass: "16px" },
      backdropBlur: { glass: "12px" },
      transitionTimingFunction: { out: "cubic-bezier(0.16, 1, 0.3, 1)" },
    },
  },
  plugins: [],
};
