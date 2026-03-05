'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const PalInterface = dynamic(
  () => import('@/components/visuals/pal-interface').then((mod) => mod.PalInterface),
  {
    ssr: false,
    loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/10" />
  }
)

export function PalSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="pal"
      title={title}
      subtitle="Human-in-the-loop annotation corrects uncertain manifolds and expands cultural ontologies beyond mainstream taxonomies."
    >
      <PalInterface />
    </SectionShell>
  )
}
