import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cream: { DEFAULT: "#FAF9F5", deep: "#F4EFE6", soft: "#FBF3E1" },
        clay: { DEFAULT: "#C96442", 600: "#B5512F", 700: "#9C4527" },
        amber: { DEFAULT: "#D9A441", soft: "#FBF3E1" },
        ink: { DEFAULT: "#1F1E1C", muted: "#6B675F" },
        line: { DEFAULT: "#E8E1D3", strong: "#DED5C3" },
      },
      fontFamily: {
        sans: [
          "-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter",
          "system-ui", "sans-serif",
        ],
        mono: ["SF Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { xl2: "1.25rem" },
      boxShadow: {
        glass: "0 1px 2px rgba(60,40,20,.04), 0 12px 40px rgba(60,40,20,.06)",
        lift: "0 2px 6px rgba(60,40,20,.06), 0 18px 48px rgba(200,100,66,.10)",
      },
      backdropBlur: { glass: "20px" },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "fade-up": { "0%": { opacity: "0", transform: "translateY(6px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
      },
      animation: {
        shimmer: "shimmer 1.5s infinite",
        "fade-up": "fade-up .25s ease both",
      },
    },
  },
  plugins: [],
};
export default config;
