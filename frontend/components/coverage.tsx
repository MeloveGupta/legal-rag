import { Reveal } from '@/components/reveal'

const panels = [
  {
    code: 'Constitution',
    name: 'The Constitution of India',
    meta: '395 Articles · 12 Schedules',
    accent: 'gold',
  },
  {
    code: 'BNS',
    name: 'Bharatiya Nyaya Sanhita',
    meta: '358 Sections · Substantive law',
    accent: 'electric',
  },
  {
    code: 'BNSS',
    name: 'Bharatiya Nagarik Suraksha Sanhita',
    meta: '531 Sections · Procedure',
    accent: 'emerald',
  },
  {
    code: 'BSA',
    name: 'Bharatiya Sakshya Adhiniyam',
    meta: '170 Sections · Evidence',
    accent: 'gold',
  },
  {
    code: 'Future Laws',
    name: 'All Indian Legislation',
    meta: 'Continuously expanding',
    accent: 'muted',
    soon: true,
  },
]

const accentGlow: Record<string, string> = {
  gold: 'hover:shadow-[0_0_60px_-15px_rgba(232,200,146,0.4)] hover:border-gold/30',
  electric:
    'hover:shadow-[0_0_60px_-15px_rgba(120,170,255,0.45)] hover:border-electric/30',
  emerald:
    'hover:shadow-[0_0_60px_-15px_rgba(104,222,178,0.4)] hover:border-emerald/30',
  muted: 'hover:border-border',
}

const accentText: Record<string, string> = {
  gold: 'text-gold',
  electric: 'text-electric',
  emerald: 'text-emerald',
  muted: 'text-muted-foreground',
}

export function Coverage() {
  return (
    <section id="coverage" className="relative px-6 py-28 md:py-40">
      <div className="mx-auto max-w-6xl">
        <Reveal className="max-w-2xl">
          <p className="text-sm uppercase tracking-[0.25em] text-gold/80">
            Legal coverage
          </p>
          <h2 className="mt-5 text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            The living corpus, connected as one graph.
          </h2>
          <p className="mt-6 text-pretty leading-relaxed text-muted-foreground">
            Every article and section exists as an interlinked node. As new
            legislation is enacted, it joins the same intelligence.
          </p>
        </Reveal>

        <div className="mt-16 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {panels.map((p, i) => (
            <Reveal key={p.code} delay={i * 80}>
              <article
                className={`glass group relative flex h-full min-h-[220px] flex-col justify-between overflow-hidden rounded-3xl p-7 transition-all duration-500 hover:-translate-y-1 ${accentGlow[p.accent]} ${
                  p.soon ? 'opacity-60' : ''
                }`}
              >
                {/* soft internal light */}
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute -right-10 -top-10 size-40 rounded-full opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-100"
                  style={{
                    background:
                      p.accent === 'electric'
                        ? 'rgba(120,170,255,0.25)'
                        : p.accent === 'emerald'
                          ? 'rgba(104,222,178,0.22)'
                          : 'rgba(232,200,146,0.22)',
                  }}
                />
                <div className="relative flex items-start justify-between">
                  <span
                    className={`font-serif text-2xl italic ${accentText[p.accent]}`}
                  >
                    {p.code}
                  </span>
                  {p.soon && (
                    <span className="rounded-full border border-border px-2.5 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                      Coming Soon
                    </span>
                  )}
                </div>
                <div className="relative">
                  <h3 className="text-lg font-medium leading-snug text-foreground">
                    {p.name}
                  </h3>
                  <p className="mt-2 font-mono text-xs text-muted-foreground">
                    {p.meta}
                  </p>
                </div>
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}
