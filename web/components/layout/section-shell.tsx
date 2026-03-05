import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'
import type { SectionId } from '@/data/copy'

type SectionShellProps = {
  id: SectionId
  title: string
  subtitle?: string
  className?: string
  children: ReactNode
}

export function SectionShell({ id, title, subtitle, className, children }: SectionShellProps) {
  return (
    <section id={id} data-section-id={id} className={cn('relative min-h-screen px-4 py-24 md:px-10', className)}>
      <div className="mx-auto max-w-7xl">
        <div className="mb-10 reveal-item">
          <span className="chapter-chip">{id}</span>
          <h2 className="mt-4 font-display text-4xl font-semibold leading-[1.02] text-textMain md:text-6xl">{title}</h2>
          {subtitle ? <p className="mt-4 max-w-3xl text-base leading-relaxed text-textSub md:text-lg">{subtitle}</p> : null}
        </div>
        {children}
      </div>
    </section>
  )
}