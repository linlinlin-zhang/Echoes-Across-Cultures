'use client'

import type { ReactNode } from 'react'

import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cn } from '@/lib/utils'
import type { SectionId } from '@/data/copy'

type SectionShellProps = {
  id: SectionId
  title: string
  subtitle?: string
  className?: string
  children: ReactNode
}

const sectionNumber: Record<SectionId, string> = {
  hero: '01',
  problem: '02',
  architecture: '03',
  galaxy: '04',
  lab: '05',
  pal: '06',
  results: '07',
  ethics: '08'
}

export function SectionShell({ id, title, subtitle, className, children }: SectionShellProps) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <section id={id} data-section-id={id} className={cn('relative min-h-screen px-4 py-24 md:px-10', className)}>
      <div className="mx-auto max-w-7xl">
        <div className="relative mb-12 reveal-item">
          <div className="mb-3 flex items-center gap-2">
            <span className="chapter-chip">{isZh ? `章节 ${sectionNumber[id]}（Chapter ${sectionNumber[id]}）` : `chapter ${sectionNumber[id]}`}</span>
            <span className="sticker">{id}</span>
          </div>
          <h2 className="max-w-5xl font-display text-4xl font-semibold leading-[0.97] text-textMain md:text-6xl">{title}</h2>
          {subtitle ? <p className="mt-4 max-w-3xl text-base leading-relaxed text-textSub md:text-lg">{subtitle}</p> : null}
        </div>
        {children}
      </div>
    </section>
  )
}
