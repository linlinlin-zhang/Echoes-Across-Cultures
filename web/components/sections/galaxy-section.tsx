'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const CultureGalaxyGraph = dynamic(() => import('@/components/visuals/culture-galaxy-graph').then((mod) => mod.CultureGalaxyGraph), {
  ssr: false,
  loading: () => <div className="h-[420px] animate-pulse rounded-3xl bg-white/80" />
})

export function GalaxySection({ title }: { title: string }) {
  return (
    <SectionShell
      id="galaxy"
      title={title}
      subtitle="A navigable culture atlas: switch alignment modes, trace OT routes, and inspect local musical grammars as connected neighborhoods."
      className="bg-[radial-gradient(circle_at_18%_14%,rgba(0,167,160,.16),transparent_34%),radial-gradient(circle_at_84%_4%,rgba(126,87,194,.14),transparent_36%)]"
    >
      <CultureGalaxyGraph />
    </SectionShell>
  )
}