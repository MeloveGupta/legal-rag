import { ChatWindow } from '@/components/chat-window'

export const metadata = {
  title: 'Pramaan - Ask Indian Law',
  description:
    'Ask questions about Indian law and get cited answers grounded in the Constitution, BNS, BNSS, and BSA.',
}

export default function ChatPage() {
  return (
    <div className="flex h-svh flex-col bg-background">
      {/* Top bar */}
      <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          {/* Back to home */}
          <a
            href="/"
            className="flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            ← Home
          </a>

          <span className="text-border">|</span>

          {/* Pramaan wordmark */}
          <a href="/" className="flex items-center gap-2">
            <span className="relative flex size-2 items-center justify-center">
              <span className="absolute size-2 rounded-full bg-gold/70 blur-[2px]" />
              <span className="relative size-1 rounded-full bg-gold" />
            </span>
            <span className="font-serif text-base leading-none tracking-tight text-foreground">
              Pramaan
            </span>
          </a>
        </div>

        {/* Status badge */}
        <div className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[10px] text-muted-foreground">
          <span className="size-1.5 rounded-full bg-emerald shadow-[0_0_5px] shadow-emerald" />
          Constitution · BNS · BNSS · BSA
        </div>
      </header>

      {/* Chat window takes remaining height */}
      <ChatWindow />
    </div>
  )
}