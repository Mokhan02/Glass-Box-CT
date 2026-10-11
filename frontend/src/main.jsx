import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { MotionConfig } from "framer-motion";
import "./index.css";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import ScanPage from "./pages/Scan";
import Why from "./pages/Why";
import About from "./pages/About";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Landing />} />
            <Route path="/scan/:id" element={<ScanPage />} />
            <Route path="/why" element={<Why />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Landing />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MotionConfig>
  </React.StrictMode>
);
