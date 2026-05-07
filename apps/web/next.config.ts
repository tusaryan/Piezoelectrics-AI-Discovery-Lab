import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Turbopack is enabled via --turbopack flag in dev script
  reactStrictMode: true,
  // Proxy API requests to FastAPI backend in dev
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/health",
        destination: "http://localhost:8000/health",
      },
      {
        source: "/ws/:path*",
        destination: "http://localhost:8000/ws/:path*",
      },
    ];
  },
};

export default nextConfig;
