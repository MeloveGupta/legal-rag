'use client'

import { useEffect, useRef, useState } from 'react'
import { Reveal } from '@/components/reveal'

const question =
  'What punishment exists under BNS for criminal intimidation?'

const answer = `Under the Bharatiya Nyaya Sanhita, 2023, criminal intimidation is defined in Section 351(1). The punishment depends on the severity of the threat:

• General threats - imprisonment up to 2 years, or a fine, or both, under Section 351(2).

• Threats to cause death or grievous hurt, destroy property by fire, or impute unchastity - imprisonment up to 7 years, or a fine, or both, under Section 351(3).

Where the intimidation is carried out anonymously, an enhanced penalty applies under Section 351(4).`

const citations = [
  { s: 'BNS §351(1)', label: 'Definition' },
  { s: 'BNS §351(2)', label: 'General' },
  { s: 'BNS §351(3)', label: 'Aggravated' },
  { s: 'BNS §351(4)', label: 'Anonymous' },
]

export function DemoConversation() {
  const [typed, setTyped] = useState('')
  const [phase, setPhase] = useState<'idle' | 'thinking' | 'typing' | 'done'>(
    'idle',
  )
  const ref = useRef<HTMLDivElement>(null)
  const started = useRef(false)

  const run = () => {
    setTyped('')
    setPhase('thinking')
    const thinkT = setTimeout(() => {
      setPhase('typing')
      let i = 0
      const interval = setInterval(() => {
        i += Math.random() > 0.85 ? 2 : 1
        setTyped(answer.slice(0, i))
        if (i >= answer.length) {
          clearInterval(interval)
          setPhase('done')
        }
      }, 14)
    }, 1100)
    return () => clearTimeout(thinkT)
  }

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting && !started.current) {
          started.current = true
          run()
        }
      },
      { threshold: 0.35 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const revealedCitations =
    phase === 'done' ? citations.length : phase === 'typing' ? Math.min(citations.length, Math.floor((typed.length / answer.length) * (citations.length + 1))) : 0

  return (
    <section id="demo" ref={ref} className="relative px-6 py-28 md:py-40">
      <div className="mx-auto max-w-5xl">
        <Reveal className="mx-auto max-w-2xl text-center">
          <p className="text-sm uppercase tracking-[0.25em] text-gold/80">
            The playground
          </p>
          <h2 className="mt-5 text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            Ask anything. See the reasoning light up.
          </h2>
        </Reveal>

        <Reveal delay={120}>
          <div className="glass mt-14 overflow-hidden rounded-3xl">
            {/* window bar */}
            <div className="flex items-center justify-between border-b border-border px-5 py-3.5">
              <div className="flex items-center gap-2">
                <span className="size-2.5 rounded-full bg-foreground/15" />
                <span className="size-2.5 rounded-full bg-foreground/15" />
                <span className="size-2.5 rounded-full bg-foreground/15" />
              </div>
              <span className="font-mono text-xs text-muted-foreground">
                pramaan · playground
              </span>
              <button
                onClick={run}
                className="rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
              >
                Replay
              </button>
            </div>

            <div className="grid gap-0 md:grid-cols-[1.4fr_1fr]">
              {/* conversation */}
              <div className="space-y-5 p-6 sm:p-8">
                {/* user */}
                <div className="flex justify-end">
                  <p className="max-w-[85%] rounded-2xl rounded-tr-sm bg-secondary px-4 py-3 text-sm leading-relaxed text-foreground">
                    {question}
                  </p>
                </div>

                {/* AI */}
                <div className="flex gap-3">
                  <span className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-gold/15 font-serif text-xs italic text-gold">
                    N
                  </span>
                  <div className="min-h-[7rem] max-w-full">
                    {phase === 'thinking' && (
                      <div className="flex items-center gap-1.5 pt-2">
                        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.2s]" />
                        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.1s]" />
                        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground" />
                      </div>
                    )}
                    {(phase === 'typing' || phase === 'done') && (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
                        {typed}
                        {phase === 'typing' && (
                          <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-electric align-middle" />
                        )}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* citations panel */}
              <div className="border-t border-border p-6 sm:p-8 md:border-l md:border-t-0">
                <p className="mb-4 text-[0.7rem] uppercase tracking-[0.2em] text-muted-foreground">
                  Grounded in
                </p>
                <ul className="space-y-2.5">
                  {citations.map((c, i) => {
                    const on = i < revealedCitations
                    return (
                      <li
                        key={c.s}
                        className={`flex items-center justify-between rounded-xl border px-3.5 py-2.5 transition-all duration-500 ${
                          on
                            ? 'border-electric/30 bg-electric/5 opacity-100'
                            : 'border-border opacity-40'
                        }`}
                      >
                        <span
                          className={`font-mono text-xs transition-colors duration-500 ${
                            on ? 'text-electric' : 'text-muted-foreground'
                          }`}
                        >
                          {c.s}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {c.label}
                        </span>
                      </li>
                    )
                  })}
                </ul>
                <p className="mt-6 text-xs leading-relaxed text-muted-foreground/70">
                  Illustrative response. Pramaan always links reasoning to the
                  exact statutory text.
                </p>
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
