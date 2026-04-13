import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Talent Scout - Interview Engine",
  description: "AI-driven interview platform for enterprise hiring",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
