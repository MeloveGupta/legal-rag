'use client'

import { useEffect, useRef, useState } from 'react'
import { Reveal } from '@/components/reveal'

const stages = [
  {
    n: '01',
    title: 'Question',
    body: 'A natural-language query, exactly as a person would ask it.',
  },
  {
    n: '02',
    title: 'Semantic Understanding',
    body: 'Intent, entities and legal concepts are parsed and disambiguated.',
  },
  {
    n: '03',
    title: 'Retrieval',
    body: 'Relevant provisions and precedents are pulled from the graph.',
  },
  {
    n: '04',
    title: 'Legal Reasoning',
    body: 'The engine weighs sections, exceptions and interpretation.',
  },
  {
    n: '05',
    title: 'Grounded Answer',
    body: 'A clear response, written in plain language you can act on.',
  },
  {
    n: '06',
    title: 'Verified Citations',
    body: 'Every claim is traced back to its exact statutory source.',
  },
]

export function Pipeline() {
  const [active, setActive] = useState(0)
  const sectionRef = useRef<HTMLDivElement>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = sectionRef.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => setInView(e.isIntersecting),
      { threshold: 0.25 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    if (!inView) return
    const id = setInterval(() => {
      setActive((a) => (a + 1) % stages.length)
    }, 1400)
    return () => clearInterval(id)
  }, [inView])

  return (
    <section id="pipeline" ref={sectionRef} className="relative px-6 py-28 md:py-40">
      <div className="mx-auto max-w-6xl">
        <Reveal>
          <p className="text-sm uppercase tracking-[0.25em] text-gold/80">
            The reasoning pipeline
          </p>
          <h2 className="mt-5 max-w-2xl text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            Watch a single question become a grounded answer.
          </h2>
        </Reveal>

        {/* Desktop horizontal flow */}
        <div className="mt-20 hidden md:block">
          <div className="relative">
            {/* base line */}
            <div className="absolute left-0 right-0 top-[13px] h-px bg-border" />
            {/* progress line */}
            <div
              className="absolute left-0 top-[13px] h-px bg-gradient-to-r from-gold via-electric to-emerald transition-all duration-700 ease-out"
              style={{
                width: `${((active + 0.5) / stages.length) * 100}%`,
                boxShadow: '0 0 12px rgba(120,170,255,0.5)',
              }}
            />
            <ol className="relative grid grid-cols-6 gap-4">
              {stages.map((s, i) => {
                const isActive = i === active
                const isPast = i < active
                return (
                  <li key={s.n} className="flex flex-col items-start">
                    <span
                      className={`relative z-10 flex size-[26px] items-center justify-center rounded-full border text-[10px] font-medium transition-all duration-500 ${
                        isActive
                          ? 'scale-110 border-transparent bg-electric text-background shadow-[0_0_20px_rgba(120,170,255,0.6)]'
                          : isPast
                            ? 'border-transparent bg-foreground/80 text-background'
                            : 'border-border bg-background text-muted-foreground'
                      }`}
                    >
                      {s.n}
                    </span>
                    <h3
                      className={`mt-5 text-sm font-medium transition-colors duration-500 ${
                        isActive ? 'text-foreground' : 'text-foreground/70'
                      }`}
                    >
                      {s.title}
                    </h3>
                    <p
                      className={`mt-2 text-xs leading-relaxed transition-opacity duration-500 ${
                        isActive ? 'text-muted-foreground opacity-100' : 'text-muted-foreground/60 opacity-70'
                      }`}
                    >
                      {s.body}
                    </p>
                  </li>
                )
              })}
            </ol>
          </div>
        </div>

        {/* Mobile vertical flow */}
        <ol className="mt-14 space-y-8 md:hidden">
          {stages.map((s, i) => (
            <li key={s.n} className="relative flex gap-4">
              {i < stages.length - 1 && (
                <span className="absolute left-[12px] top-7 h-full w-px bg-border" />
              )}
              <span
                className={`relative z-10 flex size-6 shrink-0 items-center justify-center rounded-full border text-[10px] transition-all duration-500 ${
                  i === active
                    ? 'border-transparent bg-electric text-background'
                    : 'border-border bg-background text-muted-foreground'
                }`}
              >
                {s.n}
              </span>
              <div>
                <h3 className="text-sm font-medium text-foreground">
                  {s.title}
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {s.body}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
