import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "HARNESS — Layer 1 Translator",
  description:
    "Terjemahkan permintaan bahasa awam menjadi Execution Contract. Mode Wizard of Oz untuk validasi nyata sebelum membangun agent.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="id">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  );
}
