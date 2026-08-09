/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the running sldgen-api so SSE, uploads and file
// routes behave exactly as they do in production, where the API serves dist/
// itself and everything is same-origin. `ws: false` and no buffering: an SSE
// response must not be held.
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: true },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.SLDGEN_API_URL ?? 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
