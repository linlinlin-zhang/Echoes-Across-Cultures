'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const RecommendationDemo = dynamic(() => import('@/components/visuals/recommendation-demo').then((mod) => mod.RecommendationDemo), {
  ssr: false,
  loading: () => <div className="h-[540px] animate-pulse rounded-3xl bg-white/80" />
})

export function ResultsSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="results"
      title={title}
      subtitle="A chapterized control deck comparing baseline recommendations and DDRL outputs under serendipity and fairness objectives."
      className="bg-[radial-gradient(circle_at_82%_18%,rgba(0,167,160,.12),transparent_38%)]"
    >
      <RecommendationDemo />
    </SectionShell>
  )
}