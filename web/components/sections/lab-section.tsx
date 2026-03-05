'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const DisentanglementLab = dynamic(
  () => import('@/components/visuals/disentanglement-lab').then((mod) => mod.DisentanglementLab),
  {
    ssr: false,
    loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/10" />
  }
)

export function LabSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="lab"
      title={title}
      subtitle="Tune latent controls in real time to inspect how content, style, and affect contributions reshape audio and representation views."
    >
      <DisentanglementLab />
    </SectionShell>
  )
}
