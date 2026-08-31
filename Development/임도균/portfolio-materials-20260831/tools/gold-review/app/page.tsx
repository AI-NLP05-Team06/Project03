import type { Metadata } from "next";
import GoldReviewApp from "./gold-review-app";

export const metadata: Metadata = {
  title: "KDIC Gold 검수 도구",
  description: "평가 질문과 Gold 청크를 한 화면에서 검토하고 편집합니다.",
};

export default function Home() {
  return <GoldReviewApp />;
}
