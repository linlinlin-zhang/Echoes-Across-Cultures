'use client'

import dynamic from 'next/dynamic'

import { SectionShell } from '@/components/layout/section-shell'

const DisentanglementLab = dynamic(() => import('@/components/visuals/disentanglement-lab').then((mod) => mod.DisentanglementLab), {
  ssr: false,
  loading: () => <div className="h-[520px] animate-pulse rounded-3xl bg-white/80" />
})

export function LabSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="lab"
      title={title}
      subtitle="Hands-on lab mode: tune latent sliders and watch audio spectra, affect plane, and cultural similarity respond in real time."
      className="bg-[radial-gradient(circle_at_78%_22%,rgba(255,111,97,.16),transparent_34%)]"
    >
      <DisentanglementLab />
    </SectionShell>
  )
}