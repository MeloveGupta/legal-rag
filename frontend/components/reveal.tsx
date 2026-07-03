'use client'

import { useEffect, useRef, type ElementType, type ReactNode } from 'react'

export function Reveal({
  children,
  as: Tag = 'div',
  delay = 0,
  className,
}: {
  children: ReactNode
  as?: ElementType
  delay?: number
  className?: string
}) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const target = entry.target as HTMLElement
            target.style.transitionDelay = `${delay}ms`
            target.classList.add('is-visible')
            observer.unobserve(target)
          }
        })
      },
      { threshold: 0.15, rootMargin: '0px 0px -8% 0px' },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [delay])

  return (
    <Tag ref={ref} data-reveal className={className}>
      {children}
    </Tag>
  )
}
