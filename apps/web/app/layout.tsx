import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import ThemeProvider from "@/components/layout/ThemeProvider";
import AppShell from "@/components/layout/AppShell";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "Piezo.AI — AI-Driven Piezoelectric Material Discovery",
  description:
    "AI-powered platform for predicting and optimizing lead-free piezoelectric materials. Train ML models, predict d33/tc/hardness, and discover novel compositions.",
  keywords: [
    "piezoelectric",
    "machine learning",
    "materials science",
    "KNN",
    "d33",
    "Curie temperature",
    "lead-free",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrainsMono.variable}`}>
        <ThemeProvider>
          <AppShell>{children}</AppShell>
        </ThemeProvider>
      </body>
    </html>
  );
}
