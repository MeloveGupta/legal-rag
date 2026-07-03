'use client'

import { useEffect, useRef, useState } from 'react'

const stats = [
  { value: 2400000, suffix: '+', label: 'Questions Answered' },
  { value: 14200, suffix: '', label: 'Sections Indexed' },
  { value: 6, suffix: '', label: 'Legal Sources' },
  { value: 0.8, suffix: 's', label: 'Avg. Retrieval Time', decimals: 1 },
]

function format(n: number, decimals = 0) {
  if (decimals > 0) return n.toFixed(decimals)
  return Math.round(n).toLocaleString('en-IN')
}

function Counter({
  value,
  suffix,
  decimals = 0,
  play,
}: {
  value: number
  suffix: string
  decimals?: number
  play: boolean
}) {
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    if (!play) return
    let raf = 0
    const duration = 1800
    const start = performance.now()
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      setDisplay(value * eased)
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [play, value])

  return (
    <span className="font-serif text-5xl tracking-tight text-foreground sm:text-6xl">
      {format(display, decimals)}
      <span className="text-gold">{suffix}</span>
    </span>
  )
}

export function LiveStats() {
  const ref = useRef<HTMLDivElement>(null)
  const [play, setPlay] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setPlay(true)
          obs.disconnect()
        }
      },
      { threshold: 0.3 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section className="relative px-6 py-28 md:py-40">
      <div ref={ref} className="mx-auto max-w-6xl">
        <div className="mb-16 flex flex-col items-center gap-3 text-center">
          <span className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-muted-foreground">
            <span className="size-1.5 animate-pulse rounded-full bg-emerald shadow-[0_0_8px] shadow-emerald" />
            Updated daily
          </span>
          <h2 className="text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            Intelligence, measured in the open.
          </h2>
        </div>

        <div className="grid grid-cols-2 gap-x-6 gap-y-14 lg:grid-cols-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className="flex flex-col items-center border-l border-border first:border-l-0 lg:border-l"
            >
              <Counter
                value={s.value}
                suffix={s.suffix}
                decimals={s.decimals}
                play={play}
              />
              <span className="mt-3 text-sm text-muted-foreground">
                {s.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
