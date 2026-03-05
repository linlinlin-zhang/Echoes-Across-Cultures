'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'

const RecommendationDemo = dynamic(() => import('@/components/visuals/recommendation-demo').then((mod) => mod.RecommendationDemo), {
  ssr: false,
  loading: () => <div className="h-[540px] animate-pulse rounded-3xl bg-white/80" />
})

export function ResultsSection({ title }: { title: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <SectionShell
      id="results"
      title={title}
      subtitle={
        isZh
          ? '章节化控制台（Control Deck）：在机缘巧合（Serendipity）与文化公平目标下，对比基线推荐与 DDRL 输出。'
          : 'A chapterized control deck comparing baseline recommendations and DDRL outputs under serendipity and fairness objectives.'
      }
      className="bg-[linear-gradient(180deg,rgba(26,115,232,.08),transparent_42%),radial-gradient(circle_at_12%_18%,rgba(234,67,53,.12),transparent_40%)]"
    >
      <RecommendationDemo />
    </SectionShell>
  )
}
