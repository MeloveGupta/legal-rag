'use client'

import { useEffect, useState } from 'react'

const links = [
  { label: 'Pipeline', href: '#pipeline' },
  { label: 'Coverage', href: '#coverage' },
  { label: 'Playground', href: '#demo' },
  { label: 'Why Pramaan', href: '#why' },
]

export function SiteNav() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header className="fixed inset-x-0 top-0 z-50 flex justify-center px-4 pt-4">
      <nav
        className={`flex w-full max-w-5xl items-center justify-between rounded-full px-4 py-2.5 transition-all duration-500 sm:px-6 ${
          scrolled
            ? 'glass shadow-[0_8px_40px_-12px_rgba(0,0,0,0.6)]'
            : 'border border-transparent'
        }`}
      >
        <a href="#top" className="flex items-center gap-2.5">
          <span className="relative flex size-2.5 items-center justify-center">
            <span className="absolute size-2.5 rounded-full bg-gold/70 blur-[3px]" />
            <span className="relative size-1.5 rounded-full bg-gold" />
          </span>
          <span className="font-serif text-xl leading-none tracking-tight text-foreground">
            Pramaan
          </span>
        </a>

        <div className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-muted-foreground transition-colors duration-300 hover:text-foreground"
            >
              {l.label}
            </a>
          ))}
        </div>

        <a
          href="#demo"
          className="group inline-flex items-center gap-1.5 rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-all duration-300 hover:gap-2.5 hover:brightness-110"
        >
          Launch
          <span className="transition-transform duration-300 group-hover:translate-x-0.5">
            →
          </span>
        </a>
      </nav>
    </header>
  )
}
