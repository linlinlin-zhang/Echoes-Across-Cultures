'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const RecommendationDemo = dynamic(
  () => import('@/components/visuals/recommendation-demo').then((mod) => mod.RecommendationDemo),
  {
    ssr: false,
    loading: () => <div className="h-[540px] animate-pulse rounded-3xl bg-white/10" />
  }
)

export function ResultsSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="results"
      title={title}
      subtitle="Compare baseline recommendation behavior with DDRL + OT trajectories under serendipity and fairness objectives."
    >
      <RecommendationDemo />
    </SectionShell>
  )
}
