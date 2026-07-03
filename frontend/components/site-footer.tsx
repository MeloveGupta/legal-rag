import { Reveal } from '@/components/reveal'

const columns = [
  {
    title: 'Product',
    links: ['Playground', 'Coverage', 'Pipeline', 'Changelog'],
  },
  {
    title: 'Corpus',
    links: ['Constitution', 'BNS', 'BNSS', 'BSA'],
  },
  {
    title: 'Company',
    links: ['About', 'Research', 'Careers', 'Contact'],
  },
]

export function SiteFooter() {
  return (
    <footer className="relative px-6 pb-12 pt-10">
      {/* Closing CTA */}
      <Reveal className="mx-auto max-w-6xl">
        <div className="glass relative overflow-hidden rounded-[2rem] px-8 py-20 text-center sm:px-16">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute left-1/2 top-0 size-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-40 blur-[120px]"
            style={{ background: 'rgba(232,200,146,0.25)' }}
          />
          <h2 className="relative mx-auto max-w-2xl text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            The future of legal reasoning is a question away.
          </h2>
          <p className="relative mx-auto mt-6 max-w-md text-pretty leading-relaxed text-muted-foreground">
            Step into the playground and ask Indian law anything.
          </p>
          <div className="relative mt-10">
            <a
              href="#demo"
              className="group inline-flex items-center gap-2 rounded-full bg-primary px-8 py-4 text-sm font-medium text-primary-foreground shadow-[0_0_50px_-10px_rgba(232,200,146,0.5)] transition-all duration-300 hover:gap-3"
            >
              Launch Playground
              <span className="transition-transform duration-300 group-hover:translate-x-1">
                →
              </span>
            </a>
          </div>
        </div>
      </Reveal>

      {/* Footer grid */}
      <div className="mx-auto mt-24 max-w-6xl">
        <div className="grid grid-cols-2 gap-10 border-t border-border pt-14 md:grid-cols-[1.5fr_repeat(3,1fr)]">
          <div>
            <a href="#top" className="flex items-center gap-2.5">
              <span className="size-2 rounded-full bg-gold" />
              <span className="font-serif text-xl tracking-tight text-foreground">
                Nyaya
              </span>
            </a>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted-foreground">
              AI legal intelligence for India — grounded, cited, and always
              learning.
            </p>
          </div>
          {columns.map((col) => (
            <div key={col.title}>
              <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground/70">
                {col.title}
              </h3>
              <ul className="mt-4 space-y-3">
                {col.links.map((l) => (
                  <li key={l}>
                    <a
                      href="#"
                      className="text-sm text-foreground/70 transition-colors duration-300 hover:text-foreground"
                    >
                      {l}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-14 flex flex-col items-center justify-between gap-3 border-t border-border pt-8 text-xs text-muted-foreground/60 sm:flex-row">
          <span>© {new Date().getFullYear()} Nyaya Intelligence.</span>
          <span>Informational only — not a substitute for legal advice.</span>
        </div>
      </div>
    </footer>
  )
}
