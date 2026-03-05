'use client'

import { sectionIds, type SectionId } from '@/data/copy'
import { cn } from '@/lib/utils'

type SideNavProps = {
  labels: Record<SectionId, string>
  activeSection: SectionId
  onNavigate: (id: SectionId) => void
}

export function SideNav({ labels, activeSection, onNavigate }: SideNavProps) {
  return (
    <aside className="fixed right-4 top-1/2 z-40 hidden -translate-y-1/2 xl:block" aria-label="Section progress">
      <div className="rounded-2xl panel-glass px-2 py-3">
        <div className="relative flex flex-col gap-1">
          {sectionIds.map((id) => {
            const active = activeSection === id
            return (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                aria-label={`Jump to ${labels[id]}`}
                className="group flex items-center justify-end gap-2 rounded-full px-2 py-1 transition hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs"
              >
                <span className={cn('hidden text-xs text-textSub transition group-hover:block', active && 'block text-textMain')}>
                  {labels[id]}
                </span>
                <span
                  className={cn(
                    'h-2.5 w-2.5 rounded-full border border-white/30 transition',
                    active ? 'scale-125 bg-zs shadow-[0_0_12px_rgba(78,205,196,.8)]' : 'bg-white/20 group-hover:bg-white/60'
                  )}
                />
              </button>
            )
          })}
        </div>
      </div>
    </aside>
  )
}