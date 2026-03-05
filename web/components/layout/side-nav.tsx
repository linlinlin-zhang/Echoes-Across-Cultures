'use client'

import { sectionIds, type SectionId } from '@/data/copy'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cn } from '@/lib/utils'

type SideNavProps = {
  labels: Record<SectionId, string>
  activeSection: SectionId
  onNavigate: (id: SectionId) => void
}

export function SideNav({ labels, activeSection, onNavigate }: SideNavProps) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <aside className="fixed right-4 top-1/2 z-40 hidden -translate-y-1/2 2xl:block" aria-label={isZh ? '章节进度导航' : 'Section progress'}>
      <div className="rounded-2xl panel-glass p-2">
        {sectionIds.map((id) => {
          const active = activeSection === id
          return (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              aria-label={isZh ? `跳转至 ${labels[id]}` : `Jump to ${labels[id]}`}
              className="group flex w-full items-center justify-end gap-2 rounded-full px-2 py-1 transition hover:bg-white/70"
            >
              <span className={cn('hidden text-xs text-textSub group-hover:block', active && 'block text-textMain')}>{labels[id]}</span>
              <span
                className={cn(
                  'h-2.5 w-2.5 rounded-full border border-ink/25 transition',
                  active ? 'scale-125 bg-za shadow-[0_0_0_2px_rgba(26,115,232,0.2)]' : 'bg-white group-hover:bg-ink/30'
                )}
              />
            </button>
          )
        })}
      </div>
    </aside>
  )
}
