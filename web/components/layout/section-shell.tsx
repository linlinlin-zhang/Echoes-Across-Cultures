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
        <div className="mb-8 reveal-item">
          <h2 className="font-display text-3xl font-bold tracking-tight text-textMain md:text-5xl">{title}</h2>
          {subtitle ? <p className="mt-3 max-w-3xl font-body text-lg text-textSub">{subtitle}</p> : null}
        </div>
        {children}
      </div>
    </section>
  )
}
