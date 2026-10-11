import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built to static files in frontend/dist, which the FastAPI server serves (glassbox/app/server.py).
// In `npm run dev`, /api is proxied to a locally running server.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 2500 },
});
