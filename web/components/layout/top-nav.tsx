'use client'

import { Globe, Eye, Waves, Sparkles } from 'lucide-react'

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
      <div className="mx-auto flex max-w-7xl items-center justify-between rounded-2xl panel-glass px-4 py-2">
        <button className="group flex items-center gap-2" onClick={() => onNavigate('hero')} aria-label="Go to hero section">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-zc via-zs to-za text-white">
            <Sparkles size={16} />
          </span>
          <span className="font-display text-base text-textMain">Soundscape Without Borders</span>
        </button>

        <nav className="hidden items-center gap-1 xl:flex" aria-label="Primary">
          {sectionIds.map((id) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={cn(
                'rounded-full px-3 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs',
                activeSection === id ? 'bg-ink/10 text-textMain' : 'text-textSub hover:bg-ink/5 hover:text-textMain'
              )}
            >
              {labels[id]}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-1.5">
          <button
            className="rounded-full border border-ink/20 bg-white px-3 py-1 text-xs font-semibold text-textSub transition hover:text-textMain"
            onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
            aria-label="Toggle language"
          >
            <Globe size={13} className="mr-1 inline" />
            {locale.toUpperCase()}
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              highContrast ? 'border-zc bg-zc/15 text-zc' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
            )}
            onClick={() => setHighContrast(!highContrast)}
            aria-label="Toggle high contrast"
          >
            <Eye size={13} className="mr-1 inline" />HC
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              reduceMotion ? 'border-zs bg-zs/15 text-zs' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
            )}
            onClick={() => setReduceMotion(!reduceMotion)}
            aria-label="Toggle reduced motion"
          >
            <Waves size={13} className="mr-1 inline" />RM
          </button>
        </div>
      </div>
    </header>
  )
}