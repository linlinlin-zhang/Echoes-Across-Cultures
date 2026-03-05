'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const PalInterface = dynamic(() => import('@/components/visuals/pal-interface').then((mod) => mod.PalInterface), {
  ssr: false,
  loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/80" />
})

export function PalSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="pal"
      title={title}
      subtitle="Participatory panel for uncertainty-driven annotation, concept expansion, and cognitive-justice coverage tracking."
      className="bg-[radial-gradient(circle_at_12%_18%,rgba(126,87,194,.14),transparent_35%)]"
    >
      <PalInterface />
    </SectionShell>
  )
}