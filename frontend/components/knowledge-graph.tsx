'use client'

import { useEffect, useRef } from 'react'

type Node = {
  x: number
  y: number
  vx: number
  vy: number
  r: number
  // 0 = normal particle, else index of anchor
  anchor?: {
    label: string
    color: [number, number, number]
    pulse: number
  }
}

const ANCHORS: { label: string; color: [number, number, number] }[] = [
  { label: 'CONSTITUTION', color: [232, 200, 146] }, // gold
  { label: 'BNS', color: [120, 170, 255] }, // electric blue
  { label: 'BNSS', color: [104, 222, 178] }, // emerald
  { label: 'BSA', color: [232, 200, 146] }, // gold
]

export function KnowledgeGraph({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const prefersReduced = window.matchMedia(
      '(prefers-reduced-motion: reduce)',
    ).matches

    let width = 0
    let height = 0
    let dpr = Math.min(window.devicePixelRatio || 1, 2)
    let nodes: Node[] = []
    const mouse = { x: -9999, y: -9999, active: false }
    let raf = 0
    let t = 0

    const NODE_COUNT_BASE = 90
    const LINK_DIST = 150
    const MOUSE_DIST = 200

    function resize() {
      const parent = canvas.parentElement
      width = parent?.clientWidth ?? window.innerWidth
      height = parent?.clientHeight ?? window.innerHeight
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      build()
    }

    function build() {
      const density = Math.round((width * height) / 15000)
      const count = Math.max(50, Math.min(NODE_COUNT_BASE, density))
      nodes = []
      for (let i = 0; i < count; i++) {
        nodes.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.16,
          vy: (Math.random() - 0.5) * 0.16,
          r: Math.random() * 1.3 + 0.6,
        })
      }
      // Place anchor nodes spread across the canvas
      const positions = [
        { x: width * 0.22, y: height * 0.34 },
        { x: width * 0.74, y: height * 0.28 },
        { x: width * 0.32, y: height * 0.72 },
        { x: width * 0.8, y: height * 0.68 },
      ]
      ANCHORS.forEach((a, i) => {
        const p = positions[i]
        nodes.push({
          x: p.x,
          y: p.y,
          vx: (Math.random() - 0.5) * 0.08,
          vy: (Math.random() - 0.5) * 0.08,
          r: 2.6,
          anchor: { label: a.label, color: a.color, pulse: Math.random() * Math.PI * 2 },
        })
      })
    }

    function draw() {
      t += 0.016
      ctx.clearRect(0, 0, width, height)

      // update positions
      for (const n of nodes) {
        n.x += n.vx
        n.y += n.vy
        if (n.x < 0 || n.x > width) n.vx *= -1
        if (n.y < 0 || n.y > height) n.vy *= -1
      }

      // links
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          const dx = a.x - b.x
          const dy = a.y - b.y
          const dist = Math.hypot(dx, dy)
          if (dist > LINK_DIST) continue

          let alpha = (1 - dist / LINK_DIST) * 0.16

          // Illuminate near cursor
          if (mouse.active) {
            const mx = (a.x + b.x) / 2 - mouse.x
            const my = (a.y + b.y) / 2 - mouse.y
            const md = Math.hypot(mx, my)
            if (md < MOUSE_DIST) {
              alpha += (1 - md / MOUSE_DIST) * 0.5
            }
          }

          const anchorGlow = a.anchor || b.anchor
          if (anchorGlow) {
            const c = (a.anchor ?? b.anchor)!.color
            ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${Math.min(alpha * 1.4, 0.6)})`
          } else {
            ctx.strokeStyle = `rgba(198,206,230,${Math.min(alpha, 0.7)})`
          }
          ctx.lineWidth = anchorGlow ? 0.9 : 0.6
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.stroke()
        }
      }

      // nodes
      for (const n of nodes) {
        if (n.anchor) {
          const pulse = 0.5 + 0.5 * Math.sin(t * 1.1 + n.anchor.pulse)
          const c = n.anchor.color
          const glow = 10 + pulse * 12
          ctx.shadowBlur = glow
          ctx.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},0.9)`
          ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.85})`
          ctx.beginPath()
          ctx.arc(n.x, n.y, n.r + pulse * 1.2, 0, Math.PI * 2)
          ctx.fill()
          ctx.shadowBlur = 0

          // label
          ctx.font =
            '600 10px var(--font-geist-sans), ui-sans-serif, system-ui'
          ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.55 + pulse * 0.35})`
          ctx.textAlign = 'center'
          // letter spacing manual
          const label = n.anchor.label
          ctx.save()
          ctx.translate(n.x, n.y - 14)
          let lx = -(label.length * 6) / 2
          for (const ch of label) {
            ctx.fillText(ch, lx, 0)
            lx += 6.4
          }
          ctx.restore()
        } else {
          let a = 0.35
          let size = n.r
          if (mouse.active) {
            const md = Math.hypot(n.x - mouse.x, n.y - mouse.y)
            if (md < MOUSE_DIST) {
              const f = 1 - md / MOUSE_DIST
              a += f * 0.6
              size += f * 1.2
            }
          }
          ctx.fillStyle = `rgba(210,216,235,${a})`
          ctx.beginPath()
          ctx.arc(n.x, n.y, size, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      raf = requestAnimationFrame(draw)
    }

    function onMove(e: PointerEvent) {
      const rect = canvas.getBoundingClientRect()
      mouse.x = e.clientX - rect.left
      mouse.y = e.clientY - rect.top
      mouse.active = true
    }
    function onLeave() {
      mouse.active = false
      mouse.x = -9999
      mouse.y = -9999
    }

    resize()
    if (prefersReduced) {
      // render a single static frame
      draw()
      cancelAnimationFrame(raf)
    } else {
      raf = requestAnimationFrame(draw)
    }

    window.addEventListener('resize', resize)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerleave', onLeave)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerleave', onLeave)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      aria-hidden="true"
    />
  )
}
