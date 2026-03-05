'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const CultureGalaxyGraph = dynamic(
  () => import('@/components/visuals/culture-galaxy-graph').then((mod) => mod.CultureGalaxyGraph),
  {
    ssr: false,
    loading: () => <div className="h-[420px] animate-pulse rounded-3xl bg-white/10" />
  }
)

export function GalaxySection({ title }: { title: string }) {
  return (
    <SectionShell
      id="galaxy"
      title={title}
      subtitle="Switch between affective alignment and structural alignment to see how recommendation bridges cultures via optimal transport trajectories."
      className="bg-[radial-gradient(circle_at_20%_10%,rgba(78,205,196,.16),transparent_35%),radial-gradient(circle_at_90%_0%,rgba(165,94,234,.18),transparent_34%)]"
    >
      <CultureGalaxyGraph />
    </SectionShell>
  )
}
