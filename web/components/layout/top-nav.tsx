'use client'

import { Globe, Eye, Waves, Sparkles } from 'lucide-react'

import { sectionIds, type SectionId } from '@/data/copy'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cn } from '@/lib/utils'

type TopNavProps = {
  brand: string
  labels: Record<SectionId, string>
  activeSection: SectionId
  onNavigate: (id: SectionId) => void
}

export function TopNav({ brand, labels, activeSection, onNavigate }: TopNavProps) {
  const { locale, setLocale, highContrast, setHighContrast, reduceMotion, setReduceMotion } = useAccessibility()
  const isZh = locale === 'zh'

  return (
    <header className="fixed inset-x-0 top-0 z-40 px-4 py-3 md:px-8">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 rounded-2xl panel-glass px-4 py-2.5">
        <button className="group flex items-center gap-2" onClick={() => onNavigate('hero')} aria-label={isZh ? '返回首页章节' : 'Go to hero section'}>
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-white text-za shadow-sm ring-1 ring-ink/10">
            <Sparkles size={15} />
          </span>
          <span className="font-display text-base text-textMain">{brand}</span>
        </button>

        <nav className="hidden items-center gap-1 xl:flex" aria-label={isZh ? '主导航' : 'Primary'}>
          {sectionIds.map((id) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={cn(
                'rounded-full px-3 py-1.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-za',
                activeSection === id ? 'bg-white text-textMain shadow-sm ring-1 ring-ink/10' : 'text-textSub hover:bg-white/70 hover:text-textMain'
              )}
            >
              {labels[id]}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-1.5">
          <button
            className="rounded-full border border-ink/15 bg-white/90 px-3 py-1 text-xs font-semibold text-textSub transition hover:text-textMain"
            onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
            aria-label={isZh ? '切换语言' : 'Toggle language'}
          >
            <Globe size={13} className="mr-1 inline" />
            {locale.toUpperCase()}
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              highContrast ? 'border-zc bg-zc/15 text-zc' : 'border-ink/15 bg-white/90 text-textSub hover:text-textMain'
            )}
            onClick={() => setHighContrast(!highContrast)}
            aria-label={isZh ? '切换高对比度' : 'Toggle high contrast'}
            title={isZh ? '高对比模式（High Contrast）' : 'High Contrast'}
          >
            <Eye size={13} className="mr-1 inline" />
            {isZh ? '高对比' : 'HC'}
          </button>
          <button
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-semibold transition',
              reduceMotion ? 'border-zs bg-zs/15 text-zs' : 'border-ink/15 bg-white/90 text-textSub hover:text-textMain'
            )}
            onClick={() => setReduceMotion(!reduceMotion)}
            aria-label={isZh ? '切换减动画' : 'Toggle reduced motion'}
            title={isZh ? '减少动画（Reduced Motion）' : 'Reduced Motion'}
          >
            <Waves size={13} className="mr-1 inline" />
            {isZh ? '减动效' : 'RM'}
          </button>
        </div>
      </div>
    </header>
  )
}
