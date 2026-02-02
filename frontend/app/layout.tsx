import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Dream Architect - Turn Hums Into Songs',
  description: 'Transform your hummed melodies into full AI-generated music tracks',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
