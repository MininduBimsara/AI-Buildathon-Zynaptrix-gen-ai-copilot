import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

import { ReduxProvider } from "../store/Provider";
import NavBar from "../components/NavBar";
import GalaxyBackground from "../components/GalaxyBackground";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      style={{ background: 'transparent' }}
    >
      <body className="min-h-full flex flex-col" style={{ background: 'transparent', position: 'relative' }}>
        <GalaxyBackground />
        <ReduxProvider>
          <NavBar />
          <main className="flex-1">
            {children}
          </main>
        </ReduxProvider>
      </body>
    </html>
  );
}
