'use client'

import { Globe, Eye, Sparkles, Waves } from 'lucide-react'

import { sectionIds, type SectionId } from '@/data/copy'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cn } from '@/lib/utils'

type TopNavProps = {
  labels: Record<SectionId, string>
  activeSection: SectionId
  onNavigate: (id: SectionId) => void
}

export function TopNav({ labels, activeSection, onNavigate }: TopNavProps) {
  const { locale, setLocale, highContrast, setHighContrast, reduceMotion, setReduceMotion } = useAccessibility()

  return (
    <header className="fixed inset-x-0 top-0 z-40 px-4 py-3 md:px-8">
      <div className="mx-auto flex max-w-7xl items-center justify-between rounded-full panel-glass px-4 py-2">
        <button className="group flex items-center gap-2" onClick={() => onNavigate('hero')} aria-label="Go to hero section">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-zs to-za text-abyss shadow-[0_0_16px_rgba(78,205,196,0.6)]">
            <Sparkles size={16} />
          </span>
          <span className="font-display text-sm font-semibold tracking-wide text-textMain md:text-base">Soundscape Without Borders</span>
        </button>

        <nav className="hidden items-center gap-1 lg:flex" aria-label="Primary">
          {sectionIds.map((id) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={cn(
                'rounded-full px-3 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs',
                activeSection === id ? 'bg-white/15 text-textMain' : 'text-textSub hover:bg-white/10 hover:text-textMain'
              )}
            >
              {labels[id]}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <button
            className="rounded-full border border-white/20 bg-black/25 px-3 py-1 text-xs font-semibold text-textSub transition hover:text-textMain"
            onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
            aria-label="Toggle language"
          >
            <Globe size={14} className="mr-1 inline" />
            {locale.toUpperCase()}
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              highContrast ? 'border-zc bg-zc/20 text-zc' : 'border-white/20 bg-black/25 text-textSub hover:text-textMain'
            )}
            onClick={() => setHighContrast(!highContrast)}
            aria-label="Toggle high contrast"
          >
            <Eye size={14} className="mr-1 inline" />
            HC
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              reduceMotion ? 'border-zs bg-zs/20 text-zs' : 'border-white/20 bg-black/25 text-textSub hover:text-textMain'
            )}
            onClick={() => setReduceMotion(!reduceMotion)}
            aria-label="Toggle reduced motion"
          >
            <Waves size={14} className="mr-1 inline" />
            RM
          </button>
        </div>
      </div>
    </header>
  )
}