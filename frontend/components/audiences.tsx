import Image from 'next/image'
import { Reveal } from '@/components/reveal'

const audiences = [
  {
    tag: 'For Advocates',
    title: 'Research that keeps pace with your practice.',
    body: 'Surface the exact provision, cross-reference related sections, and build arguments grounded in current law, in the time it takes to read a headnote.',
    img: '/abstract-advocates.png',
    alt: 'Abstract network of glowing lines converging to a bright focal point',
  },
  {
    tag: 'For Law Students',
    title: 'Understand the why, not just the what.',
    body: 'Trace how a principle connects across statutes, see the reasoning laid out step by step, and learn the new criminal codes as one coherent system.',
    img: '/abstract-students.png',
    alt: 'Abstract branching constellation of connected particle nodes',
  },
  {
    tag: 'For Every Citizen',
    title: 'The law, in language you actually speak.',
    body: 'Ask a plain question about your rights and get a clear, cited answer, no jargon, no gatekeeping, no assumptions about what you already know.',
    img: '/abstract-citizens.png',
    alt: 'Abstract concentric ripples of light emanating from a bright core',
  },
]

export function Audiences() {
  return (
    <section id="why" className="relative px-6 py-28 md:py-40">
      <div className="mx-auto max-w-6xl">
        <Reveal className="max-w-2xl">
          <p className="text-sm uppercase tracking-[0.25em] text-gold/80">
            Why people use Pramaan
          </p>
          <h2 className="mt-5 text-balance font-serif text-4xl leading-tight tracking-tight text-foreground sm:text-5xl">
            One intelligence, understood differently by everyone.
          </h2>
        </Reveal>

        <div className="mt-20 flex flex-col gap-24 md:gap-32">
          {audiences.map((a, i) => (
            <div
              key={a.tag}
              className={`grid items-center gap-10 md:grid-cols-2 md:gap-16 ${
                i % 2 === 1 ? 'md:[&>*:first-child]:order-2' : ''
              }`}
            >
              <Reveal>
                <div className="relative aspect-[4/3] overflow-hidden rounded-3xl border border-border">
                  <Image
                    src={a.img || '/placeholder.svg'}
                    alt={a.alt}
                    fill
                    sizes="(max-width: 768px) 100vw, 50vw"
                    className="object-cover"
                  />
                  <div
                    aria-hidden="true"
                    className="absolute inset-0"
                    style={{
                      background:
                        'radial-gradient(circle at 50% 50%, transparent 40%, rgba(9,9,11,0.5) 100%)',
                    }}
                  />
                </div>
              </Reveal>

              <Reveal delay={100}>
                <div>
                  <span className="font-serif text-lg italic text-gold">
                    {a.tag}
                  </span>
                  <h3 className="mt-4 text-balance font-serif text-3xl leading-tight tracking-tight text-foreground sm:text-4xl">
                    {a.title}
                  </h3>
                  <p className="mt-6 max-w-md text-pretty leading-relaxed text-muted-foreground">
                    {a.body}
                  </p>
                </div>
              </Reveal>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
