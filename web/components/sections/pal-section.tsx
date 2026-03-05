'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'

const PalInterface = dynamic(() => import('@/components/visuals/pal-interface').then((mod) => mod.PalInterface), {
  ssr: false,
  loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/80" />
})

export function PalSection({ title }: { title: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <SectionShell
      id="pal"
      title={title}
      subtitle={
        isZh
          ? '参与式面板（Participatory Panel）：基于不确定性驱动标注、扩展文化概念并追踪认知正义覆盖度。'
          : 'Participatory panel for uncertainty-driven annotation, concept expansion, and cognitive-justice coverage tracking.'
      }
      className="bg-[linear-gradient(180deg,rgba(24,128,56,.07),transparent_42%),radial-gradient(circle_at_84%_24%,rgba(26,115,232,.1),transparent_44%)]"
    >
      <PalInterface />
    </SectionShell>
  )
}
