import React from "react";
import { createRoot } from "react-dom/client";
import GoldReviewApp from "./app/gold-review-app";
import "./app/globals.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("앱을 표시할 영역을 찾을 수 없습니다.");
}

createRoot(root).render(
  <React.StrictMode>
    <GoldReviewApp />
  </React.StrictMode>,
);
