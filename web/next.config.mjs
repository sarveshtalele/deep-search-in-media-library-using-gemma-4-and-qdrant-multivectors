/** @type {import('next').NextConfig} */
// Use 127.0.0.1 (not "localhost") so Node doesn't try IPv6 ::1 first — uvicorn
// binds IPv4 only, and the ::1 attempt floods the log with ECONNREFUSED.
const API = process.env.DEEPSEARCH_API_URL || "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* to the FastAPI backend so the frontend stays same-origin
  // (media/thumbnail URLs returned as /api/... just work in <video>/<img>).
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
