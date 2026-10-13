import { defineConfig } from "vite";

// Local development only: the Vite dev server proxies MedVaani API and audio
// routes to the FastAPI backend (uvicorn app.main:app on 127.0.0.1:8000), so
// the browser stays same-origin and no CORS changes are needed.
//
// Note: Explicit target+changeOrigin+secure prevents some Vite 8 edge cases
// where the shorthand proxy can fail to forward matching routes in certain
// environments.
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/voice": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
        xfwd: false,
      },
      "/audio": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
        xfwd: false,
      },
    },
  },
});
