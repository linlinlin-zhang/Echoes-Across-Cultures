'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDownRight, Layers3, Pointer, Workflow } from 'lucide-react'

import { cultureNodes, otDemoRoutes, songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { PulseConsole } from '@/components/visuals/pulse-console'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp, cn } from '@/lib/utils'

type GalleryMode = 'observe' | 'disentangle' | 'transport'
type FactorState = Record<'zc' | 'zs' | 'za', boolean>

const LatentCinemaStage = dynamic(() => import('@/components/visuals/latent-cinema-stage').then((mod) => mod.LatentCinemaStage), {
  ssr: false,
  loading: () => <div className="h-[58vh] animate-pulse rounded-3xl bg-white/80" />
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
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)
  const setSeparation = useSceneStore((state) => state.setSeparation)

  const [galleryMode, setGalleryMode] = useState<GalleryMode>('observe')
  const [cultureFilter, setCultureFilter] = useState('All')
  const [routeIndex, setRouteIndex] = useState(0)
  const energy = 0.64
  const [sceneStep, setSceneStep] = useState(0)
  const [autoScene, setAutoScene] = useState(false)
  const [factorState, setFactorState] = useState<FactorState>({ zc: true, zs: true, za: true })

  const hoveredSong = useMemo(() => songPoints.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? null, [hoveredSongId, selectedSongId])

  const scenes = useMemo(
    () =>
      isZh
        ? [
            { name: '内容', mode: 'observe' as const, factors: { zc: true, zs: false, za: false } },
            { name: '文化', mode: 'observe' as const, factors: { zc: false, zs: true, za: false } },
            { name: '情感', mode: 'observe' as const, factors: { zc: false, zs: false, za: true } },
            { name: '迁移', mode: 'transport' as const, factors: { zc: true, zs: true, za: true } },
            { name: '混合', mode: 'disentangle' as const, factors: { zc: true, zs: true, za: true } }
          ]
        : [
            { name: 'Content', mode: 'observe' as const, factors: { zc: true, zs: false, za: false } },
            { name: 'Culture', mode: 'observe' as const, factors: { zc: false, zs: true, za: false } },
            { name: 'Affect', mode: 'observe' as const, factors: { zc: false, zs: false, za: true } },
            { name: 'Transfer', mode: 'transport' as const, factors: { zc: true, zs: true, za: true } },
            { name: 'Blend', mode: 'disentangle' as const, factors: { zc: true, zs: true, za: true } }
          ],
    [isZh]
  )

  useEffect(() => {
    const scene = scenes[sceneStep] ?? scenes[0]
    setGalleryMode(scene.mode)
    setFactorState(scene.factors)
  }, [sceneStep, scenes])

  useEffect(() => {
    if (!autoScene) return
    const timer = window.setInterval(() => {
      setSceneStep((prev) => (prev + 1) % scenes.length)
    }, 2800)
    return () => window.clearInterval(timer)
  }, [autoScene, scenes.length])

  useEffect(() => {
    if (galleryMode !== 'transport') return
    const timer = window.setInterval(() => {
      setRouteIndex((prev) => (prev + 1) % otDemoRoutes.length)
    }, 3200)
    return () => window.clearInterval(timer)
  }, [galleryMode])

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setSeparation(clamp(value * 1.25, 0, 1))
  })

  const routeCultureNames = useMemo(() => {
    const idToName = new Map(cultureNodes.map((node) => [node.id, node.name]))
    const route = otDemoRoutes[routeIndex] ?? otDemoRoutes[0]
    return route.map((id) => idToName.get(id) ?? id)
  }, [routeIndex])

  const toggleFactor = (factor: keyof FactorState) => {
    setFactorState((prev) => {
      const next = { ...prev, [factor]: !prev[factor] }
      if (Object.values(next).some(Boolean)) return next
      return prev
    })
  }

  const signatureCards = isZh
    ? [
        { key: 'ddrl', title: 'DDRL', caption: '三因子解纠缠', tone: 'from-zc/14 to-zc/5 border-zc/30 text-zc' },
        { key: 'ot', title: 'OT', caption: '跨文化迁移', tone: 'from-zs/14 to-zs/5 border-zs/30 text-zs' },
        { key: 'pal', title: 'PAL', caption: '人机反馈闭环', tone: 'from-za/14 to-za/5 border-za/30 text-za' }
      ]
    : [
        { key: 'ddrl', title: 'DDRL', caption: 'Tri-factor disentanglement', tone: 'from-zc/14 to-zc/5 border-zc/30 text-zc' },
        { key: 'ot', title: 'OT', caption: 'Cross-cultural transfer', tone: 'from-zs/14 to-zs/5 border-zs/30 text-zs' },
        { key: 'pal', title: 'PAL', caption: 'Human-in-the-loop feedback', tone: 'from-za/14 to-za/5 border-za/30 text-za' }
      ]

  return (
    <section id="hero" ref={sectionRef} data-section-id="hero" className="relative min-h-screen overflow-hidden px-4 pb-10 pt-28 md:px-10" aria-labelledby="hero-title">
      <div className="mx-auto max-w-7xl">
        <div className="mb-6 flex flex-wrap items-center gap-2 reveal-item">
          <span className="sticker">{isZh ? '策展实验（Curated Experiment）' : 'curated experiment'}</span>
          <span className="sticker">{isZh ? '交互叙事（Interactive Narrative）' : 'interactive narrative'}</span>
          <span className="sticker">{isZh ? '跨文化推荐（Cross-Cultural Recsys）' : 'cross-cultural recsys'}</span>
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.84fr_1.16fr]">
          <motion.div className="reveal-item space-y-5" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65 }}>
            <h1 id="hero-title" className="hero-title-glow font-display text-5xl font-semibold leading-[0.9] text-textMain md:text-7xl xl:text-8xl">
              {title}
            </h1>
            <p className="max-w-xl text-base leading-relaxed text-textSub md:text-lg">{lead}</p>

            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-zc/30 bg-zc/10 px-3 py-1 text-zc">zc</span>
              <span className="rounded-full border border-zs/30 bg-zs/10 px-3 py-1 text-zs">zs</span>
              <span className="rounded-full border border-za/30 bg-za/10 px-3 py-1 text-za">za</span>
            </div>

            <div className="grid gap-2 md:grid-cols-3">
              {signatureCards.map((item) => (
                <div key={item.key} className={cn('rounded-2xl border bg-gradient-to-b p-3', item.tone)}>
                  <p className="font-display text-xl text-textMain">{item.title}</p>
                  <p className="mt-1 text-xs text-textSub">{item.caption}</p>
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
                <span className="font-mono uppercase tracking-[0.1em]">{isZh ? '提示' : 'note'}</span>
              </div>
              <p className="mt-1 text-sm text-textSub">{hint}</p>
            </div>

            {hoveredSong ? (
              <div className="max-w-xl">
                <SongRadar song={hoveredSong} />
                {selectedSongId ? (
                  <button className="mt-2 text-xs text-textSub underline decoration-dotted underline-offset-4" onClick={() => setSelectedSongId(null)}>
                    {isZh ? '清除已选' : 'Clear selected'}
                  </button>
                ) : null}
              </div>
            ) : (
              <div className="max-w-xl rounded-2xl border border-ink/15 bg-white/85 px-4 py-3 text-sm text-textSub">{isZh ? '点击舞台中的任一点查看歌曲解剖卡。' : 'Click any point in the stage to inspect a song anatomy card.'}</div>
            )}
          </motion.div>

          <div className="reveal-item space-y-4">
            <div className="relative overflow-hidden rounded-3xl paper-card">
              <div className="absolute left-3 top-3 z-10 chapter-chip">{isZh ? '潜空间动态舞台' : 'latent motion stage'}</div>
              <div className="absolute right-3 top-3 z-10 sticker">{galleryMode.toUpperCase()}</div>

              <div className="h-[58vh] md:h-[64vh]">
                <LatentCinemaStage galleryMode={galleryMode} cultureFilter={cultureFilter} factorState={factorState} routeIndex={routeIndex} energy={energy} sceneStep={sceneStep} />
              </div>

              <div className="absolute inset-x-3 bottom-3 z-10 rounded-2xl border border-ink/15 bg-white/86 p-2.5">
                <div className="mb-2 flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    {scenes.map((scene, index) => (
                      <button
                        key={`${scene.name}-${index}`}
                        onClick={() => setSceneStep(index)}
                        className={cn('h-6 min-w-6 rounded-full border px-2 text-[11px] font-semibold transition', sceneStep === index ? 'border-za/40 bg-za/10 text-za' : 'border-ink/20 bg-white text-textSub')}
                      >
                        {index + 1}
                      </button>
                    ))}
                  </div>
                  <button onClick={() => setAutoScene((prev) => !prev)} className={cn('rounded-full border px-2.5 py-1 text-[11px] font-semibold', autoScene ? 'border-zs/35 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub')}>
                    {isZh ? (autoScene ? '自动中' : '自动') : autoScene ? 'AUTO ON' : 'AUTO'}
                  </button>
                </div>

                <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-center">
                  <div className="rounded-xl border border-ink/12 bg-white px-2.5 py-1.5 text-xs text-textSub">{scenes[sceneStep]?.name}</div>

                  <select value={cultureFilter} onChange={(event) => setCultureFilter(event.target.value)} className="rounded-xl border border-ink/20 bg-white px-2 py-1.5 text-xs text-textMain">
                    <option value="All">{isZh ? '全部文化' : 'All'}</option>
                    {Array.from(new Set(songPoints.map((item) => item.culture))).map((culture) => (
                      <option key={culture} value={culture}>
                        {culture}
                      </option>
                    ))}
                  </select>

                  <button onClick={() => setRouteIndex((prev) => (prev + 1) % otDemoRoutes.length)} className="inline-flex items-center rounded-xl border border-ink/20 bg-white px-2.5 py-1.5 text-xs font-semibold text-textSub">
                    <Workflow size={12} className="mr-1" />
                    {isZh ? '路径' : 'Route'}
                  </button>
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  {(['zc', 'zs', 'za'] as const).map((factor) => (
                    <button
                      key={factor}
                      onClick={() => toggleFactor(factor)}
                      className={cn(
                        'inline-flex items-center rounded-full border px-2 py-1 text-[11px] font-semibold transition',
                        factorState[factor]
                          ? factor === 'zc'
                            ? 'border-zc/35 bg-zc/10 text-zc'
                            : factor === 'zs'
                              ? 'border-zs/35 bg-zs/10 text-zs'
                              : 'border-za/35 bg-za/10 text-za'
                          : 'border-ink/20 bg-white text-textSub'
                      )}
                    >
                      <Layers3 size={11} className="mr-1" />
                      {factor}
                    </button>
                  ))}

                  <span className="ml-auto rounded-full border border-ink/15 bg-white px-2 py-1 text-[11px] text-textSub">{routeCultureNames.join(' -> ')}</span>
                </div>
              </div>
            </div>

            <PulseConsole />
          </div>
        </div>
      </div>

      <p className="sr-only">{isZh ? '潜空间展厅已切换为动态分镜舞台。' : 'Latent gallery is now a dynamic storyboard stage.'}</p>
    </section>
  )
}
