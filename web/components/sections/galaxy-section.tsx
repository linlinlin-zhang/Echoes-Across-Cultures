'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'

const CultureGalaxyGraph = dynamic(() => import('@/components/visuals/culture-galaxy-graph').then((mod) => mod.CultureGalaxyGraph), {
  ssr: false,
  loading: () => <div className="h-[420px] animate-pulse rounded-3xl bg-white/80" />
})

export function GalaxySection({ title }: { title: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <SectionShell
      id="galaxy"
      title={title}
      subtitle={
        isZh
          ? '可导航的文化图谱：切换对齐模式（Alignment Mode）、追踪最优传输路径（Optimal Transport Route），并查看地方音乐语法的连接邻域。'
          : 'A navigable culture atlas: switch alignment modes, trace OT routes, and inspect local musical grammars as connected neighborhoods.'
      }
      className="bg-[linear-gradient(180deg,rgba(26,115,232,.08),transparent_42%),radial-gradient(circle_at_82%_18%,rgba(24,128,56,.14),transparent_38%)]"
    >
      <CultureGalaxyGraph />
    </SectionShell>
  )
}
