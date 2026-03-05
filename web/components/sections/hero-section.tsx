'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useMemo, useRef } from 'react'
import { ArrowDownRight, Keyboard, Orbit, Pointer, ScrollText } from 'lucide-react'

import { songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { PulseConsole } from '@/components/visuals/pulse-console'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useAccessibility } from '@/components/providers/accessibility-provider'

const LatentSpaceCanvas = dynamic(() => import('@/components/visuals/latent-space-canvas').then((mod) => mod.LatentSpaceCanvas), {
  ssr: false,
  loading: () => <div className="h-[56vh] animate-pulse rounded-3xl bg-white/80" />
})

type HeroSectionProps = {
  title: string
  lead: string
  hint: string
  ctaPrimary: string
  ctaSecondary: string
  onNavigate: (id: 'galaxy' | 'lab') => void
}

export function HeroSection({ title, lead, hint, ctaPrimary, ctaSecondary, onNavigate }: HeroSectionProps) {
  const sectionRef = useRef<HTMLElement>(null)
  const isMobile = useMediaQuery('(max-width: 767px)')
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)
  const setSeparation = useSceneStore((state) => state.setSeparation)

  const hoveredSong = useMemo(() => songPoints.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? null, [hoveredSongId, selectedSongId])

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setSeparation(Math.min(1, value * 1.35))
  })

  const capabilityCards = isZh
    ? [
        { icon: ScrollText, title: '叙事滚动（Narrate）', text: '按编辑式章节滚动浏览研究叙事。' },
        { icon: Orbit, title: '轨道旋转（Orbit）', text: '在 3D 中旋转潜空间星群并观察因子分离。' },
        { icon: Keyboard, title: '键盘触发（Play）', text: '用键盘音垫触发实时声学脉冲。' }
      ]
    : [
        { icon: ScrollText, title: 'Narrate', text: 'Scroll chapters as an editorial story.' },
        { icon: Orbit, title: 'Orbit', text: 'Rotate latent constellations in 3D.' },
        { icon: Keyboard, title: 'Play', text: 'Trigger sonic pulses with keyboard pads.' }
      ]

  return (
    <section id="hero" ref={sectionRef} data-section-id="hero" className="relative min-h-screen overflow-hidden px-4 pb-10 pt-28 md:px-10" aria-labelledby="hero-title">
      <div className="mx-auto max-w-7xl">
        <div className="mb-6 flex flex-wrap items-center gap-2 reveal-item">
          <span className="sticker">{isZh ? '策展实验（Curated Experiment）' : 'curated experiment'}</span>
          <span className="sticker">{isZh ? '交互叙事（Interactive Narrative）' : 'interactive narrative'}</span>
          <span className="sticker">{isZh ? '跨文化推荐（Cross-Cultural Recsys）' : 'cross-cultural recsys'}</span>
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <motion.div className="reveal-item space-y-5" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65 }}>
            <h1 id="hero-title" className="hero-title-glow font-display text-5xl font-semibold leading-[0.9] text-textMain md:text-7xl xl:text-8xl">
              {title}
            </h1>
            <p className="max-w-xl text-base leading-relaxed text-textSub md:text-lg">{lead}</p>

            <div className="grid gap-2 md:grid-cols-3">
              {capabilityCards.map((item) => (
                <div key={item.title} className="note-card">
                  <item.icon size={14} className="text-za" />
                  <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.09em] text-textSub">{item.title}</p>
                  <p className="mt-1 text-sm text-textSub">{item.text}</p>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-3 pt-1">
              <button onClick={() => onNavigate('galaxy')} className="group rounded-full bg-za px-5 py-2.5 font-semibold text-white transition hover:-translate-y-0.5 hover:shadow-glow">
                {ctaPrimary}
                <ArrowDownRight size={16} className="ml-2 inline transition group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
              </button>
              <button onClick={() => onNavigate('lab')} className="rounded-full border border-ink/20 bg-white px-5 py-2.5 font-semibold text-textMain transition hover:border-za/45 hover:bg-za/5">
                {ctaSecondary}
              </button>
            </div>

            <div className="rounded-2xl border border-ink/15 bg-white/85 p-3">
              <div className="flex items-center gap-2 text-xs text-textSub">
                <Pointer size={12} />
                <span className="font-mono uppercase tracking-[0.1em]">{isZh ? '交互提示（Interaction Note）' : 'interaction note'}</span>
              </div>
              <p className="mt-1 text-sm text-textSub">{hint}</p>
            </div>

            {hoveredSong ? (
              <div className="max-w-xl">
                <SongRadar song={hoveredSong} />
                {selectedSongId ? (
                  <button className="mt-2 text-xs text-textSub underline decoration-dotted underline-offset-4" onClick={() => setSelectedSongId(null)}>
                    {isZh ? '清除已选解剖视图（Clear Selected Anatomy View）' : 'Clear selected anatomy view'}
                  </button>
                ) : null}
              </div>
            ) : null}
          </motion.div>

          <div className="reveal-item space-y-4">
            <div className="relative overflow-hidden rounded-3xl paper-card scanline">
              <div className="absolute left-3 top-3 z-10 chapter-chip">{isZh ? '潜空间展厅（Latent Space Gallery）' : 'latent space gallery'}</div>
              <div className="h-[56vh] md:h-[62vh]">
                {isMobile ? (
                  <div className="flex h-full items-center justify-center px-6 text-center">
                    <p className="max-w-md text-base text-textSub">
                      {isZh
                        ? '移动端为简化预览模式（Mobile Preview Mode）。桌面端可解锁完整 3D 交互与点位检视（Full 3D Interaction and Point Inspection）。'
                        : 'Mobile mode shows a simplified preview. Desktop unlocks full 3D interaction and point inspection.'}
                    </p>
                  </div>
                ) : (
                  <LatentSpaceCanvas />
                )}
              </div>
            </div>
            <PulseConsole />
          </div>
        </div>
      </div>

      <p className="sr-only">
        {isZh
          ? '三维潜空间地图展示内容、风格与情感三条解纠缠轨迹（Three-dimensional latent map with disentangled content, style, and affect trajectories）。'
          : 'Three-dimensional latent map with disentangled content, style, and affect trajectories.'}
      </p>
    </section>
  )
}
