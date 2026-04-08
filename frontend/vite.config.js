import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const apiProxyTarget = "http://127.0.0.1:8011";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
