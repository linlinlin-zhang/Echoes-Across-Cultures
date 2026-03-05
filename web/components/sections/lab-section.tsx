'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'

const DisentanglementLab = dynamic(() => import('@/components/visuals/disentanglement-lab').then((mod) => mod.DisentanglementLab), {
  ssr: false,
  loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/80" />
})

export function LabSection({ title }: { title: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <SectionShell
      id="lab"
      title={title}
      subtitle={
        isZh
          ? '实验室模式（Lab Mode）：调节潜变量滑杆，实时观察频谱、情感平面与文化相似度如何联动变化。'
          : 'Hands-on lab mode: tune latent sliders and watch audio spectra, affect plane, and cultural similarity respond in real time.'
      }
      className="bg-[linear-gradient(180deg,rgba(234,67,53,.08),transparent_40%),radial-gradient(circle_at_18%_78%,rgba(26,115,232,.12),transparent_44%)]"
    >
      <DisentanglementLab />
    </SectionShell>
  )
}
