import { SiteNav } from '@/components/site-nav'
import { Hero } from '@/components/hero'
import { Pipeline } from '@/components/pipeline'
import { Coverage } from '@/components/coverage'
import { DemoConversation } from '@/components/demo-conversation'
import { Audiences } from '@/components/audiences'
import { LiveStats } from '@/components/live-stats'
import { SiteFooter } from '@/components/site-footer'

export default function Page() {
  return (
    <main className="relative min-h-screen overflow-x-hidden bg-background">
      <SiteNav />
      <Hero />
      <Pipeline />
      <Coverage />
      <DemoConversation />
      <Audiences />
      <LiveStats />
      <SiteFooter />
    </main>
  )
}
