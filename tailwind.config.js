const c = (name) => `rgb(var(--${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: c("bg"),
        surface: c("surface"),
        raised: c("raised"),
        line: c("line"),
        fg: c("fg"),
        muted: c("muted"),
        accent: c("accent"),
        ok: c("ok"),
        warn: c("warn"),
        bad: c("bad"),
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
