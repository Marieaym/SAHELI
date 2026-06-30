/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        night: "rgb(var(--c-night) / <alpha-value>)",
        surface: "rgb(var(--c-surface) / <alpha-value>)",
        card: "rgb(var(--c-card) / <alpha-value>)",
        cardBorder: "rgb(var(--c-cardBorder) / <alpha-value>)",
        gold: "rgb(var(--c-gold) / <alpha-value>)",
        goldBright: "rgb(var(--c-goldBright) / <alpha-value>)",
        sand: "rgb(var(--c-sand) / <alpha-value>)",
        muted: "rgb(var(--c-muted) / <alpha-value>)",
        clay: "rgb(var(--c-clay) / <alpha-value>)",
        amber: "rgb(var(--c-amber) / <alpha-value>)",
        acacia: "rgb(var(--c-acacia) / <alpha-value>)",
        primary: "rgb(var(--c-primary) / <alpha-value>)",
        indigo: "rgb(var(--c-indigo) / <alpha-value>)",
        sidebar: "rgb(var(--c-sidebar) / <alpha-value>)",
      },
      fontFamily: {
        display: ["Fraunces", "serif"],
        body: ["Inter", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      borderRadius: {
        "4xl": "2rem",
        "5xl": "2.5rem",
      },
      boxShadow: {
        bento: "0 8px 40px rgba(15, 23, 42, 0.08), 0 2px 8px rgba(15, 23, 42, 0.04)",
        "bento-lg": "0 20px 60px rgba(15, 23, 42, 0.12), 0 4px 16px rgba(15, 23, 42, 0.06)",
        glow: "0 0 40px rgba(var(--c-primary), 0.15)",
      },
    },
  },
  plugins: [],
};
