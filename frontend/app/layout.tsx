import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono, Instrument_Serif } from 'next/font/google'
import './globals.css'

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] })
const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})
const instrumentSerif = Instrument_Serif({
  variable: '--font-instrument-serif',
  subsets: ['latin'],
  weight: '400',
  style: ['normal', 'italic'],
})

export const metadata: Metadata = {
  title: 'Pramaan - AI Legal Intelligence for India',
  description:
    'Understand Indian law as clearly as it was written. Ask questions in natural language and receive answers grounded directly in the Constitution, BNS, BNSS, and BSA, with citations and legal reasoning.',
  generator: 'v0.app',
  keywords: [
    'Indian law',
    'AI legal intelligence',
    'BNS',
    'BNSS',
    'BSA',
    'Constitution of India',
    'legal research',
  ],
}

export const viewport: Viewport = {
  colorScheme: 'dark',
  themeColor: '#09090B',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${instrumentSerif.variable} bg-background`}
    >
      <body className="font-sans antialiased">
        {children}
        <div className="noise-overlay" aria-hidden="true" />
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}
