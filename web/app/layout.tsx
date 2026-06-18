import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Search · Media Library",
  description:
    "Cross-modal natural-language search over local video, audio & text — powered by Gemma 4 and Qdrant.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
