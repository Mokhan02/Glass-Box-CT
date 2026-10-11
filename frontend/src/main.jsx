import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { MotionConfig } from "framer-motion";
import "./index.css";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import ScanPage from "./pages/Scan";
import Evidence from "./pages/Evidence";
import About from "./pages/About";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/scan/:id" element={<ScanPage />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="/why" element={<Navigate to="/evidence" replace />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Landing />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MotionConfig>
  </React.StrictMode>
);
