import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5034',
        changeOrigin: true,
        secure: false,
      }
    },
    host: '127.0.0.1',
    port: parseInt(process.env.PORT || '3000'),
    allowedHosts: true // You can also specify domains like ['yourdomain.com', 'another.com']
  }
})
