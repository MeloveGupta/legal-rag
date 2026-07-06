'use client'

import { useEffect, useRef, useState } from 'react'
import { queryLegalRAG, type Source } from '@/lib/api'
import ReactMarkdown from 'react-markdown'

// ── Types ──────────────────────────────────────────────────────────────────

interface UserMessage {
  role: 'user'
  id: string
  text: string
}

interface AIMessage {
  role: 'ai'
  id: string
  text: string
  sources: Source[]
  answered: boolean
  latency_ms: number
  streaming: boolean   // true while the typewriter is still running
}

type Message = UserMessage | AIMessage

// ── Starter prompts ────────────────────────────────────────────────────────

const STARTERS = [
  'What does Article 21 say about the right to life?',
  'What is the procedure for amendment of the Constitution?',
  'What are the rights of an arrested person under BNSS?',
  'What punishment exists under BNS for criminal intimidation?',
  'What does Article 32 say about constitutional remedies?',
  'What does Article 44 say about a Uniform Civil Code?',
]

// ── Helpers ────────────────────────────────────────────────────────────────

function sourceLabel(src: Source) {
  return src.file_name
    .replace('.pdf', '')
    .replace(/_/g, ' ')
    .replace('constitution of india', 'Constitution')
    .replace('bns', 'BNS')
    .replace('bnss', 'BNSS')
    .replace('bsa', 'BSA')
}

function uid() {
  return Math.random().toString(36).slice(2)
}

// ── Citation accordion ─────────────────────────────────────────────────────

function Citations({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (sources.length === 0) return null

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-[0.7rem] uppercase tracking-[0.18em] text-electric/70 transition-colors hover:text-electric"
      >
        <span
          className={`inline-block transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
        >
          ›
        </span>
        {sources.length} source{sources.length > 1 ? 's' : ''} cited
      </button>

      {open && (
        <ul className="mt-2.5 space-y-1.5">
          {sources.map((src, i) => (
            <li
              key={`${src.file_name}-${src.chunk_index}`}
              className="flex flex-col rounded-xl border border-electric/20 bg-electric/5 px-3.5 py-2.5"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-electric">
                  {sourceLabel(src)} · p.{src.page}
                </span>
                <span className="text-[10px] text-muted-foreground">[{i + 1}]</span>
              </div>
              <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-muted-foreground/70">
                {src.content_preview}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── Single message bubbles ─────────────────────────────────────────────────

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[78%] rounded-2xl rounded-tr-sm bg-secondary px-4 py-3 text-sm leading-relaxed text-foreground">
        {text}
      </p>
    </div>
  )
}

function AIBubble({ msg }: { msg: AIMessage }) {
  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-gold/15 font-serif text-xs italic text-gold">
        N
      </span>

      <div className="min-w-0 flex-1">
        {msg.streaming && msg.text === '' ? (
          /* Thinking dots */
          <div className="flex items-center gap-1.5 pt-2">
            <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.2s]" />
            <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.1s]" />
            <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground" />
          </div>
        ) : (
          <>
            {msg.streaming ? (
              /* Plain text while streaming — partial markdown would render broken mid-tag */
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
                {msg.text}
                <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-electric align-middle" />
              </p>
            ) : (
              /* Full markdown render once answer is complete */
              <div className="text-sm leading-relaxed text-foreground/90 [&_strong]:font-semibold [&_strong]:text-foreground [&_ol]:mt-2 [&_ol]:list-decimal [&_ol]:space-y-3 [&_ol]:pl-5 [&_ul]:mt-2 [&_ul]:list-disc [&_ul]:space-y-2 [&_ul]:pl-5 [&_li]:leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            )}

            {!msg.streaming && (
              <>
                <Citations sources={msg.sources} />
                <p className="mt-2 text-[10px] text-muted-foreground/40">
                  {msg.latency_ms > 0 ? `${(msg.latency_ms / 1000).toFixed(1)}s` : ''}
                </p>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── Empty state ────────────────────────────────────────────────────────────

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-16 text-center">
      {/* Logo mark */}
      <span className="relative mb-6 flex size-12 items-center justify-center">
        <span className="absolute size-12 rounded-full bg-gold/20 blur-xl" />
        <span className="relative flex size-8 items-center justify-center rounded-full border border-gold/30 bg-gold/10">
          <span className="font-serif text-lg italic text-gold">N</span>
        </span>
      </span>

      <h2 className="font-serif text-2xl tracking-tight text-foreground">
        What would you like to know?
      </h2>
      <p className="mt-2 max-w-sm text-sm text-muted-foreground">
        Ask anything about the Constitution of India, BNS, BNSS, or BSA.
        Every answer cites the exact statute it came from.
      </p>

      <div className="mt-8 grid w-full max-w-lg gap-2 sm:grid-cols-2">
        {STARTERS.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="rounded-xl border border-border bg-card/30 px-4 py-3 text-left text-xs leading-snug text-muted-foreground transition-all duration-200 hover:border-border/60 hover:bg-card/60 hover:text-foreground"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Main chat window ───────────────────────────────────────────────────────

export function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (question: string) => {
    const q = question.trim()
    if (!q || busy) return

    setBusy(true)
    setInput('')

    // 1. Append user message
    const userMsg: UserMessage = { role: 'user', id: uid(), text: q }
    const aiId = uid()
    const placeholderAI: AIMessage = {
      role: 'ai',
      id: aiId,
      text: '',
      sources: [],
      answered: true,
      latency_ms: 0,
      streaming: true,
    }

    setMessages((prev) => [...prev, userMsg, placeholderAI])

    try {
      // 2. Call the backend
      const res = await queryLegalRAG(q)
      const fullText = res.answer

      // 3. Typewriter effect
      let i = 0
      const tick = () => {
        i += Math.random() > 0.85 ? 2 : 1
        const slice = fullText.slice(0, i)
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aiId ? { ...m, text: slice, streaming: true } as AIMessage : m,
          ),
        )
        if (i < fullText.length) {
          setTimeout(tick, 14)
        } else {
          // Typewriter done — set final state with sources
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiId
                ? ({
                    ...m,
                    text: fullText,
                    sources: res.sources,
                    answered: res.answered,
                    latency_ms: res.latency_ms,
                    streaming: false,
                  } as AIMessage)
                : m,
            ),
          )
          setBusy(false)
          inputRef.current?.focus()
        }
      }
      tick()
    } catch (err) {
      const errText =
        err instanceof Error
          ? err.message
          : 'Something went wrong. Is the backend running?'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId
            ? ({
                ...m,
                text: errText,
                sources: [],
                answered: false,
                latency_ms: 0,
                streaming: false,
              } as AIMessage)
            : m,
        ),
      )
      setBusy(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    send(input)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex h-full flex-col">
      {/* Message area */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <EmptyState onSelect={send} />
        ) : (
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <UserBubble key={msg.id} text={msg.text} />
              ) : (
                <AIBubble key={msg.id} msg={msg} />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input bar — fixed to bottom */}
      <div className="border-t border-border bg-background/80 backdrop-blur-sm">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex w-full max-w-3xl items-end gap-2 px-4 py-4"
        >
          <div className="flex flex-1 items-center gap-2 rounded-2xl border border-border bg-card/40 px-4 py-3 focus-within:border-electric/40 transition-colors">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Indian law…"
              disabled={busy}
              autoFocus
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none disabled:opacity-50"
            />
            {busy && (
              <span className="flex shrink-0 items-center gap-1">
                <span className="size-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                <span className="size-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.07s]" />
                <span className="size-1 animate-bounce rounded-full bg-muted-foreground" />
              </span>
            )}
          </div>
          <button
            type="submit"
            disabled={!input.trim() || busy}
            className="shrink-0 rounded-xl bg-primary px-4 py-3 text-xs font-medium text-primary-foreground transition-opacity disabled:opacity-40"
          >
            Ask
          </button>
        </form>
        <p className="pb-3 text-center text-[10px] text-muted-foreground/40">
          Grounded in Constitution of India · BNS · BNSS · BSA
        </p>
      </div>
    </div>
  )
}