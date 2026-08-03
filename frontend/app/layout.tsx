import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Basic RAG - Learn Retrieval-Augmented Generation",
  description:
    "A minimal, explainable RAG demo: ingest documents, then ask questions and get answers with sources.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
