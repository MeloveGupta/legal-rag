import { KnowledgeGraph } from '@/components/knowledge-graph'

const trustedBy = ['Law Students', 'Advocates', 'Researchers', 'Citizens']

export function Hero() {
  return (
    <section
      id="top"
      className="relative flex min-h-[100svh] flex-col items-center justify-center overflow-hidden px-6 pb-24 pt-32"
    >
      {/* Living knowledge network */}
      <div className="absolute inset-0 z-0">
        <KnowledgeGraph className="size-full" />
      </div>

      {/* Radial vignette to focus the center */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-10"
        style={{
          background:
            'radial-gradient(ellipse 70% 60% at 50% 42%, transparent 0%, rgba(9,9,11,0.55) 62%, rgba(9,9,11,0.92) 100%)',
        }}
      />

      <div className="relative z-20 mx-auto flex max-w-4xl flex-col items-center text-center">
        <div
          data-reveal
          className="is-visible mb-8 inline-flex items-center gap-2 rounded-full border border-border bg-card/40 px-4 py-1.5 text-xs tracking-wide text-muted-foreground backdrop-blur-sm"
          style={{ animation: 'fade-up 1s ease both' }}
        >
          <span className="size-1.5 rounded-full bg-emerald shadow-[0_0_8px] shadow-emerald" />
          AI Legal Intelligence for India
        </div>

        <h1
          className="text-balance font-serif text-5xl font-normal leading-[1.05] tracking-tight text-foreground sm:text-6xl md:text-7xl lg:text-[5.25rem]"
          style={{ animation: 'fade-up 1.1s ease 0.08s both' }}
        >
          Understand Indian Law
          <br />
          <span className="italic text-ivory glow-gold">
            as clearly as it was written.
          </span>
        </h1>

        <p
          className="mt-8 max-w-xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg"
          style={{ animation: 'fade-up 1.1s ease 0.18s both' }}
        >
          Ask questions in natural language. Receive answers grounded directly
          in Indian law, with precise citations and transparent legal
          reasoning.
        </p>

        <div
          className="mt-10 flex flex-col items-center gap-4 sm:flex-row"
          style={{ animation: 'fade-up 1.1s ease 0.28s both' }}
        >
          <a
            href="#demo"
            className="group inline-flex items-center gap-2 rounded-full bg-primary px-7 py-3.5 text-sm font-medium text-primary-foreground shadow-[0_0_40px_-8px_rgba(232,200,146,0.4)] transition-all duration-300 hover:gap-3 hover:shadow-[0_0_50px_-6px_rgba(232,200,146,0.55)]"
          >
            Launch Playground
            <span className="transition-transform duration-300 group-hover:translate-x-1">
              →
            </span>
          </a>
          <a
            href="#coverage"
            className="inline-flex items-center gap-2 rounded-full border border-border px-7 py-3.5 text-sm font-medium text-foreground transition-all duration-300 hover:border-foreground/30 hover:bg-card/40"
          >
            See Coverage
          </a>
        </div>

        <div
          className="mt-20 flex flex-col items-center gap-4"
          style={{ animation: 'fade-up 1.2s ease 0.42s both' }}
        >
          <span className="text-[0.7rem] uppercase tracking-[0.25em] text-muted-foreground/70">
            Trusted by
          </span>
          <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2">
            {trustedBy.map((t) => (
              <span
                key={t}
                className="font-serif text-lg italic text-foreground/75"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
