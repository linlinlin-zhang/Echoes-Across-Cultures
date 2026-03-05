'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDownRight, Filter, Keyboard, Layers3, Orbit, Pointer, ScrollText, Workflow } from 'lucide-react'

import { cultureNodes, otDemoRoutes, songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { PulseConsole } from '@/components/visuals/pulse-console'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp, cn } from '@/lib/utils'

type GalleryMode = 'observe' | 'disentangle' | 'transport'
type FactorState = Record<'zc' | 'zs' | 'za', boolean>

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

  const [galleryMode, setGalleryMode] = useState<GalleryMode>('observe')
  const [cultureFilter, setCultureFilter] = useState('All')
  const [routeIndex, setRouteIndex] = useState(0)
  const [energy, setEnergy] = useState(0.62)
  const [manualSeparation, setManualSeparation] = useState(0.42)
  const [factorState, setFactorState] = useState<FactorState>({ zc: true, zs: true, za: true })
  const [scrollSeparation, setScrollSeparation] = useState(0)

  const hoveredSong = useMemo(() => songPoints.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? null, [hoveredSongId, selectedSongId])

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setScrollSeparation(clamp(value * 1.2, 0, 1))
  })

  useEffect(() => {
    const modeBias = galleryMode === 'observe' ? 0.05 : galleryMode === 'disentangle' ? 0.22 : 0.38
    const fusion = clamp(scrollSeparation * 0.48 + manualSeparation * 0.52 + modeBias * energy, 0, 1)
    setSeparation(fusion)
  }, [energy, galleryMode, manualSeparation, scrollSeparation, setSeparation])

  useEffect(() => {
    if (galleryMode !== 'transport') return
    const timer = window.setInterval(() => {
      setRouteIndex((prev) => (prev + 1) % otDemoRoutes.length)
    }, 3200)
    return () => window.clearInterval(timer)
  }, [galleryMode])

  const capabilityCards = isZh
    ? [
        { icon: ScrollText, title: '叙事滚动（Narrate）', text: '按编辑式章节滚动浏览研究叙事。' },
        { icon: Orbit, title: '空间巡航（Orbit）', text: '切换模式后观察潜空间几何重排。' },
        { icon: Keyboard, title: '实时演奏（Play）', text: '键盘音垫与展厅状态联动，实时反馈。' }
      ]
    : [
        { icon: ScrollText, title: 'Narrate', text: 'Scroll chapters as an editorial story.' },
        { icon: Orbit, title: 'Orbit', text: 'Switch modes and inspect latent geometry reconfiguration.' },
        { icon: Keyboard, title: 'Play', text: 'Keyboard pads react to gallery states in real-time.' }
      ]

  const cultureOptions = useMemo(() => ['All', ...Array.from(new Set(songPoints.map((item) => item.culture)))], [])

  const nodeNameById = useMemo(() => new Map(cultureNodes.map((node) => [node.id, node.name])), [])
  const activeRoute = otDemoRoutes[routeIndex] ?? otDemoRoutes[0]
  const activeRouteNames = useMemo(() => activeRoute.map((id) => nodeNameById.get(id) ?? id), [activeRoute, nodeNameById])

  const enabledFactorCount = useMemo(() => Object.values(factorState).filter(Boolean).length, [factorState])

  const missions = useMemo(
    () => [
      {
        id: 'm1',
        label: isZh ? '选择一个非默认文化筛选（Set a non-default culture filter）' : 'Set a non-default culture filter',
        done: cultureFilter !== 'All'
      },
      {
        id: 'm2',
        label: isZh ? '切到迁移模式并保持至少两个因子激活（Transport mode + 2 active factors）' : 'Transport mode + at least 2 active factors',
        done: galleryMode === 'transport' && enabledFactorCount >= 2
      },
      {
        id: 'm3',
        label: isZh ? '点击粒子打开歌曲解剖卡（Inspect one song card）' : 'Inspect one song anatomy card',
        done: Boolean(selectedSongId)
      }
    ],
    [cultureFilter, enabledFactorCount, galleryMode, isZh, selectedSongId]
  )

  const missionProgress = missions.filter((item) => item.done).length / missions.length

  const toggleFactor = (factor: keyof FactorState) => {
    setFactorState((prev) => {
      const next = { ...prev, [factor]: !prev[factor] }
      if (Object.values(next).some(Boolean)) return next
      return prev
    })
  }

  const modeOptions: Array<{ key: GalleryMode; labelZh: string; labelEn: string }> = [
    { key: 'observe', labelZh: '观测态', labelEn: 'Observe' },
    { key: 'disentangle', labelZh: '解纠缠态', labelEn: 'Disentangle' },
    { key: 'transport', labelZh: '迁移态', labelEn: 'Transport' }
  ]

  const visualLegend = useMemo(() => {
    const modeNote =
      galleryMode === 'observe'
        ? isZh
          ? '观测态：三类点云保持接近，主要用于看原始结构。'
          : 'Observe: three factor clouds stay relatively mixed for structural inspection.'
        : galleryMode === 'disentangle'
          ? isZh
            ? '解纠缠态：三类点云会拉开，便于识别内容/文化/情感边界。'
            : 'Disentangle: three clouds separate to expose content/culture/affect boundaries.'
          : isZh
            ? '迁移态：绿色路径表示 OT 偏好迁移路线，路径上的文化会被强调。'
            : 'Transport: green route shows OT preference transfer path, with route cultures highlighted.'

    return {
      modeNote,
      rows: isZh
        ? [
            { swatch: 'bg-zc', text: '红色点：内容因子 zc（旋律/节奏）' },
            { swatch: 'bg-zs', text: '绿色点：文化因子 zs（语法/乐器）' },
            { swatch: 'bg-za', text: '蓝色点：情感因子 za（效价/唤醒）' },
            { swatch: 'bg-ink/25', text: '细直线：潜空间因子方向轴（不是推荐边）' },
            { swatch: 'bg-zs', text: '绿色折线路径：跨文化 OT 迁移轨道' },
            { swatch: 'bg-white border border-ink/35', text: '环形光圈：各因子的聚类中心提示' }
          ]
        : [
            { swatch: 'bg-zc', text: 'Red dots: content factor zc (melody/rhythm)' },
            { swatch: 'bg-zs', text: 'Green dots: culture factor zs (grammar/instrumentation)' },
            { swatch: 'bg-za', text: 'Blue dots: affect factor za (valence/arousal)' },
            { swatch: 'bg-ink/25', text: 'Thin axis lines: latent factor directions (not recommendation edges)' },
            { swatch: 'bg-zs', text: 'Green route lines: OT cross-cultural transfer path' },
            { swatch: 'bg-white border border-ink/35', text: 'Halo rings: factor cluster center hints' }
          ]
    }
  }, [galleryMode, isZh])

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

            <div className="rounded-2xl border border-ink/15 bg-white/88 p-4">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <p className="chapter-chip">{isZh ? '潜空间任务台（Latent Mission Deck）' : 'latent mission deck'}</p>
                <span className="sticker">{isZh ? `完成度 ${(missionProgress * 100).toFixed(0)}%` : `${(missionProgress * 100).toFixed(0)}% completed`}</span>
              </div>

              <div className="grid gap-2 md:grid-cols-3">
                {modeOptions.map((mode) => (
                  <button
                    key={mode.key}
                    onClick={() => setGalleryMode(mode.key)}
                    className={cn(
                      'rounded-xl border px-3 py-2 text-xs font-semibold transition',
                      galleryMode === mode.key ? 'border-za/40 bg-za/10 text-za' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
                    )}
                  >
                    {isZh ? `${mode.labelZh}（${mode.labelEn}）` : mode.labelEn}
                  </button>
                ))}
              </div>

              <div className="mt-3 grid gap-2 md:grid-cols-[1fr_1fr_auto] md:items-end">
                <label className="block">
                  <span className="mb-1 block text-xs text-textSub">{isZh ? '文化筛选（Culture Filter）' : 'Culture Filter'}</span>
                  <div className="relative">
                    <Filter size={13} className="pointer-events-none absolute left-2.5 top-2.5 text-textSub" />
                    <select value={cultureFilter} onChange={(event) => setCultureFilter(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white py-2 pl-8 pr-2 text-sm text-textMain">
                      {cultureOptions.map((culture) => (
                        <option key={culture} value={culture}>
                          {culture === 'All' ? (isZh ? '全部文化（All Cultures）' : 'All Cultures') : culture}
                        </option>
                      ))}
                    </select>
                  </div>
                </label>

                <label className="block">
                  <span className="mb-1 block text-xs text-textSub">{isZh ? '传输路径（Transport Route）' : 'Transport Route'}</span>
                  <div className="rounded-xl border border-ink/20 bg-white px-3 py-2 text-xs text-textSub">{activeRouteNames.join('  ->  ')}</div>
                </label>

                <button onClick={() => setRouteIndex((prev) => (prev + 1) % otDemoRoutes.length)} className="inline-flex items-center rounded-xl border border-ink/20 bg-white px-3 py-2 text-xs font-semibold text-textSub hover:text-textMain">
                  <Workflow size={13} className="mr-1" />
                  {isZh ? '切换路径' : 'Next Route'}
                </button>
              </div>

              <div className="mt-3 grid gap-2 md:grid-cols-3">
                {([
                  { key: 'zc', labelZh: '内容因子', labelEn: 'Content' },
                  { key: 'zs', labelZh: '文化因子', labelEn: 'Culture' },
                  { key: 'za', labelZh: '情感因子', labelEn: 'Affect' }
                ] as const).map((item) => (
                  <button
                    key={item.key}
                    onClick={() => toggleFactor(item.key)}
                    className={cn('inline-flex items-center justify-center rounded-xl border px-3 py-2 text-xs font-semibold transition', factorState[item.key] ? 'border-zs/40 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub')}
                  >
                    <Layers3 size={12} className="mr-1" />
                    {isZh ? `${item.labelZh}（${item.labelEn}）` : item.labelEn}
                  </button>
                ))}
              </div>

              <div className="mt-3 grid gap-2 md:grid-cols-2">
                <label className="block">
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{isZh ? '展厅能量（Gallery Energy）' : 'Gallery Energy'}</span>
                    <span>{energy.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.01} value={energy} onChange={(event) => setEnergy(Number(event.target.value))} className="w-full accent-zs" />
                </label>
                <label className="block">
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{isZh ? '分离强度（Separation Bias）' : 'Separation Bias'}</span>
                    <span>{manualSeparation.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.01} value={manualSeparation} onChange={(event) => setManualSeparation(Number(event.target.value))} className="w-full accent-za" />
                </label>
              </div>

              <div className="mt-3 space-y-1.5 text-xs text-textSub">
                {missions.map((mission) => (
                  <p key={mission.id} className={mission.done ? 'text-zs' : ''}>
                    {mission.done ? '●' : '○'} {mission.label}
                  </p>
                ))}
              </div>
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
              <div className="absolute right-3 top-3 z-10 sticker">{isZh ? `${galleryMode} 模式（Mode）` : `${galleryMode} mode`}</div>
              <div className="h-[56vh] md:h-[62vh]">
                {isMobile ? (
                  <div className="relative flex h-full flex-col items-center justify-center gap-4 overflow-hidden px-6 text-center">
                    <div className="absolute -left-10 top-10 h-40 w-40 rounded-full bg-zc/20 blur-2xl" />
                    <div className="absolute -right-12 bottom-8 h-44 w-44 rounded-full bg-zs/20 blur-2xl" />
                    <div className="absolute left-1/2 top-1/2 h-48 w-48 -translate-x-1/2 -translate-y-1/2 rounded-full border border-za/30" />
                    <div className="relative rounded-2xl border border-ink/15 bg-white/85 px-4 py-3">
                      <p className="text-sm text-textSub">
                        {isZh
                          ? '移动端是“展厅速览”（Mobile Exhibition Quick View）。可切模式、筛选文化并查看任务进度；桌面端提供完整 3D 操作。'
                          : 'Mobile offers a quick exhibition view: switch modes, filter cultures, and track missions; desktop unlocks full 3D interaction.'}
                      </p>
                    </div>
                  </div>
                ) : (
                  <LatentSpaceCanvas galleryMode={galleryMode} cultureFilter={cultureFilter} activeFactors={factorState} routeIndex={routeIndex} energy={energy} />
                )}
              </div>
            </div>

            <div className="rounded-3xl border border-ink/15 bg-white/88 p-4">
              <p className="chapter-chip">{isZh ? '图例说明（Visual Legend）' : 'visual legend'}</p>
              <p className="mt-2 text-sm text-textSub">{visualLegend.modeNote}</p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {visualLegend.rows.map((item) => (
                  <div key={item.text} className="flex items-start gap-2 rounded-xl border border-ink/12 bg-white px-2.5 py-2 text-xs text-textSub">
                    <span className={cn('mt-0.5 inline-block h-2.5 w-2.5 rounded-full', item.swatch)} />
                    <span>{item.text}</span>
                  </div>
                ))}
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
