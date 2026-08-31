import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KDIC Gold 검수 도구",
  description: "KDIC RAG 평가데이터셋 Gold 청크 검수 도구",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
