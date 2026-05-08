import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from "path";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api/v1": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"), // Map '@' to 'src'
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            // Group Chart.js together (Heavy)
            if (id.includes("chart.js") || id.includes("vue-chartjs")) {
              return "vendor-charts";
            }
            // Group PrimeVue together (Heavy)
            if (id.includes("primevue")) {
              return "vendor-primevue";
            }
            // Group core Vue framework (Router, i18n, etc.)
            if (
              id.includes("vue") ||
              id.includes("vue-router") ||
              id.includes("vue-i18n")
            ) {
              return "vendor-vue";
            }
            // Everything else (Lucide, js-cookie, etc.)
            return "vendor-others";
          }
        },
      },
    },
  },
});