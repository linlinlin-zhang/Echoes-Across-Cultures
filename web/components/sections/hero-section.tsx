'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { motion, useMotionValueEvent, useScroll, AnimatePresence } from 'framer-motion'
import { ArrowDownRight, Sparkles, Play, Pause, Info } from 'lucide-react'

import { songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { buildFactorMetrics } from '@/lib/factor-mapping'
import { cn } from '@/lib/utils'
import { SectionShell } from '@/components/layout/section-shell'

// Direct imports for organic components
import { OrganicLatentNebula } from '@/components/visuals/organic-latent-nebula'
import { OrganicAudioGarden } from '@/components/visuals/organic-audio-garden'
import { LatentExplorer } from '@/components/visuals/latent-explorer'
import { OrganicFactorPads } from '@/components/visuals/organic-factor-pads'

type ViewMode = 'immersive' | 'classic'

interface HeroSectionProps {
  title: string
  lead: string
  hint: string
  ctaPrimary: string
  ctaSecondary: string
  onNavigate: (id: 'galaxy' | 'lab') => void
}

// Factor card with organic styling - unified with site style
function FactorCard({
  factor,
  title,
  subtitle,
  value,
  color,
  description,
  isActive,
  onClick
}: {
  factor: 'zc' | 'zs' | 'za'
  title: string
  subtitle: string
  value: number
  color: string
  description: string
  isActive: boolean
  onClick: () => void
}) {
  return (
    <motion.button
      onClick={onClick}
      className={cn(
        'group relative overflow-hidden rounded-2xl border p-4 text-left transition-all duration- min-h-[80px]',
        isActive
          ? 'border-za/40 bg-white shadow-[0_10px_26px_rgba(26,115,232,0.14)]'
          : 'paper-card hover:border-ink/35'
      )}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Progress bar background */}
      <div
        className="absolute bottom-0 left-0 top-0 opacity-10 transition-all group-hover:opacity-15"
        style={{ width: `${value * 100}%`, backgroundColor: color }}
      />

      {/* Content */}
      <div className="relative">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-medium" style={{ color }}>{factor}</span>
            <h3 className="mt-1 font-display text-lg font-semibold text-textMain">{title}</h3>
          </div>
          <div
            className="flex h-10 w-10 items-center justify-center rounded-full"
            style={{ backgroundColor: `${color}15`, border: `1px solid ${color}30` }}
          >
            <span className="font-mono text-sm font-bold" style={{ color }}>
              {Math.round(value * 100)}%
            </span>
          </div>
        </div>

        <p className="mt-2 text-xs text-textSub">{subtitle}</p>

        {/* Expandable description */}
        <AnimatePresence>
          {isActive && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 text-[11px] leading-relaxed text-textSub"
            >
              {description}
            </motion.p>
          )}
        </AnimatePresence>

        {/* Active indicator */}
        <motion.div
          className="absolute bottom-0 left-0 h-0.5"
          style={{ backgroundColor: color }}
          initial={{ width: '0%' }}
          animate={{ width: isActive ? '100%' : '0%' }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </motion.button>
  )
}

// Organic stat bar - unified style
function OrganicStatBar({ label, value, color, icon: Icon }: { label: string; value: number; color: string; icon: React.ComponentType<{ size?: number | string; style?: React.CSSProperties }> }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className="flex h-8 w-8 items-center justify-center rounded-full"
        style={{ backgroundColor: `${color}15` }}
      >
        <Icon size={14} style={{ color }} />
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-textSub">{label}</span>
          <span className="font-mono" style={{ color }}>{Math.round(value * 100)}%</span>
        </div>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink/10">
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundColor: color }}
            initial={{ width: 0 }}
            animate={{ width: `${value * 100}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>
      </div>
    </div>
  )
}

export function HeroSection({ title, lead, hint, ctaPrimary, ctaSecondary, onNavigate }: HeroSectionProps) {
  const sectionRef = useRef<HTMLElement>(null)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const setSeparation = useSceneStore((state) => state.setSeparation)

  const [viewMode, setViewMode] = useState<ViewMode>('immersive')
  const [activeFactor, setActiveFactor] = useState<'zc' | 'zs' | 'za' | null>(null)
  const [showInfo, setShowInfo] = useState(false)
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)

  const [cultureFilter] = useState('All')
  const [sceneStep, setSceneStep] = useState(0)

  const filteredSongs = useMemo(() => {
    const pool = cultureFilter === 'All' ? songPoints : songPoints.filter((song) => song.culture === cultureFilter)
    return pool.length > 0 ? pool : songPoints
  }, [cultureFilter])

  const focusedSong = useMemo(() => filteredSongs.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? filteredSongs[0], [filteredSongs, hoveredSongId, selectedSongId])
  const focusedMetrics = useMemo(() => buildFactorMetrics(focusedSong), [focusedSong])

  const scenes = useMemo(
    () =>
      isZh
        ? [{ name: '内容' }, { name: '文化' }, { name: '情感' }, { name: '迁移' }, { name: '混合' }]
        : [{ name: 'Content' }, { name: 'Culture' }, { name: 'Affect' }, { name: 'Transfer' }, { name: 'Blend' }],
    [isZh]
  )


  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setSeparation(Math.min(1, value * 1.25))
  })

  // Auto-play demo
  useEffect(() => {
    if (!isAutoPlaying) return
    const interval = setInterval(() => {
      setSceneStep(prev => (prev + 1) % scenes.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [isAutoPlaying, scenes.length])

  return (
    <SectionShell
      id="hero"
      title={title}
      subtitle={lead}
      className="min-h-screen bg-[linear-gradient(180deg,rgba(255,255,255,0.35),transparent_35%)]"
    >
      <div className="grid gap-4 xl:grid-cols-[320px_1fr_320px] xl:items-start">
        {/* Left panel - Factor cards + Audio Garden */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col gap-4"
        >
          {/* Factor cards */}
          <div className="flex flex-col gap-3">
            <FactorCard
              factor="zc"
              title={isZh ? '内容' : 'Content'}
              subtitle={isZh ? '旋律与节奏维度' : 'Melody & Rhythm'}
              value={focusedMetrics.zcStrength}
              color="#ea4335"
              description={isZh
                ? 'zc 编码音乐的旋律轮廓、节奏密度和和声进行。这是音乐最本质的叙事层。'
                : 'zc encodes melodic contour, rhythmic density, and harmonic progression—the narrative core of music.'}
              isActive={activeFactor === 'zc'}
              onClick={() => setActiveFactor(activeFactor === 'zc' ? null : 'zc')}
            />
            <FactorCard
              factor="zs"
              title={isZh ? '文化' : 'Culture'}
              subtitle={isZh ? '文化与音色维度' : 'Culture & Timbre'}
              value={focusedMetrics.zsStrength}
              color="#188038"
              description={isZh
                ? 'zs 捕捉文化风格、乐器音色和表演语境。这是音乐的"指纹"。'
                : 'zs captures cultural style, instrumental timbre, and performance context—the fingerprint of music.'}
              isActive={activeFactor === 'zs'}
              onClick={() => setActiveFactor(activeFactor === 'zs' ? null : 'zs')}
            />
            <FactorCard
              factor="za"
              title={isZh ? '情感' : 'Affect'}
              subtitle={isZh ? '效价与唤醒维度' : 'Valence & Arousal'}
              value={(focusedMetrics.zaArousal + 1) / 2}
              color="#1a73e8"
              description={isZh
                ? 'za 表征情感效价(正负)和唤醒度(强弱)。这是音乐的情感引力。'
                : 'za represents emotional valence and arousal—the gravitational pull of music.'}
              isActive={activeFactor === 'za'}
              onClick={() => setActiveFactor(activeFactor === 'za' ? null : 'za')}
            />
          </div>

          {/* Current song stats */}
          <div className="paper-card rounded-2xl p-3.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wider text-textSub">
                {isZh ? '当前样本' : 'Current Specimen'}
              </span>
              <span className="sticker truncate max-w-[120px]">{focusedSong.title}</span>
            </div>

            <div className="mt-2.5 space-y-2.5">
              <OrganicStatBar
                label={isZh ? '内容密度' : 'Content Density'}
                value={focusedMetrics.rhythmDensity}
                color="#ea4335"
                icon={Sparkles}
              />
              <OrganicStatBar
                label={isZh ? '文化强度' : 'Cultural Intensity'}
                value={focusedMetrics.zsStrength}
                color="#188038"
                icon={Sparkles}
              />
              <OrganicStatBar
                label={isZh ? '情感唤醒' : 'Emotional Arousal'}
                value={(focusedMetrics.zaArousal + 1) / 2}
                color="#1a73e8"
                icon={Sparkles}
              />
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onNavigate('galaxy')}
              className="group flex items-center gap-1.5 rounded-full bg-za px-4 py-2 text-xs font-medium text-white transition hover:bg-za/90"
            >
              {ctaPrimary}
              <ArrowDownRight size={12} className="transition group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
            </button>
            <button
              onClick={() => onNavigate('lab')}
              className="rounded-full border border-ink/20 bg-white px-4 py-2 text-xs font-medium text-textMain transition hover:border-za/40 hover:bg-za/5"
            >
              {ctaSecondary}
            </button>
          </div>

          {/* Audio Garden - 移到左侧，高度与右侧对齐 */}
          <OrganicAudioGarden className="h-[220px]" />
        </motion.div>

        {/* Center - Main visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="relative"
        >
          {/* View mode toggle */}
          <div className="mb-3 flex items-center justify-between">
            <div className="flex gap-1 rounded-full border border-ink/20 bg-white p-1">
              <button
                onClick={() => setViewMode('immersive')}
                className={cn(
                  'rounded-full px-3 py-1.5 text-xs transition',
                  viewMode === 'immersive'
                    ? 'bg-za/10 text-za'
                    : 'text-textSub hover:text-textMain'
                )}
              >
                {isZh ? '沉浸' : 'Immersive'}
              </button>
              <button
                onClick={() => setViewMode('classic')}
                className={cn(
                  'rounded-full px-3 py-1.5 text-xs transition',
                  viewMode === 'classic'
                    ? 'bg-za/10 text-za'
                    : 'text-textSub hover:text-textMain'
                )}
              >
                {isZh ? '经典' : 'Classic'}
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowInfo(!showInfo)}
                className="flex h-8 w-8 items-center justify-center rounded-full border border-ink/20 bg-white text-textSub transition hover:bg-za/5 hover:text-za"
              >
                <Info size={14} />
              </button>
              <button
                onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                className="flex h-8 w-8 items-center justify-center rounded-full border border-ink/20 bg-white text-textSub transition hover:bg-za/5 hover:text-za"
              >
                {isAutoPlaying ? <Pause size={14} /> : <Play size={14} />}
              </button>
            </div>
          </div>

          {viewMode === 'immersive' ? (
            <OrganicLatentNebula className="h-[736px] w-full" />
          ) : (
            <div className="relative h-[736px] overflow-hidden rounded-3xl border border-ink/15 bg-white">
              <div className="flex h-full items-center justify-center text-textSub">
                {isZh ? '经典模式开发中' : 'Classic mode coming soon'}
              </div>
            </div>
          )}

          {/* Scene selector */}
          <div className="mt-4 flex items-center justify-center gap-2">
            {scenes.map((scene, index) => (
              <button
                key={scene.name}
                onClick={() => setSceneStep(index)}
                className={cn(
                  'h-2 rounded-full transition-all',
                  sceneStep === index
                    ? 'w-8 bg-za'
                    : 'w-2 bg-ink/20 hover:bg-ink/40'
                )}
              />
            ))}
          </div>
        </motion.div>

        {/* Right panel - Explorer & Factor Pads */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-col gap-4"
        >
          {/* Latent Explorer - 增加高度确保完整展示 */}
          <LatentExplorer className="h-[500px]" />

          {/* Factor Pads - 增加高度确保完整展示 */}
          <OrganicFactorPads className="h-[300px]" />
        </motion.div>
      </div>

      {/* Info panel overlay */}
      <AnimatePresence>
        {showInfo && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-8 left-1/2 z-50 w-full max-w-2xl -translate-x-1/2 px-4"
          >
            <div className="rounded-2xl border border-ink/20 bg-white/95 p-6 shadow-2xl backdrop-blur-xl">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-display text-lg font-semibold text-textMain">
                    {isZh ? '潜空间序章' : 'Prelude to Latent Space'}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-textSub">
                    {isZh
                      ? '欢迎来到因子星云——一个可视化的跨文化音乐潜空间。在这里，每首歌曲都被编码为三个维度的向量：zc（内容）、zs（文化）、za（情感）。通过探索这个三维空间，你可以直观理解深度解纠缠是如何工作的。'
                      : 'Welcome to the Factor Nebula—a visualization of cross-cultural music latent space. Here, each song is encoded as a vector across three dimensions: zc (content), zs (culture), za (affect). Explore this 3D space to understand how deep disentanglement works.'}
                  </p>
                </div>
                <button
                  onClick={() => setShowInfo(false)}
                  className="ml-4 rounded-full p-1 text-textSub transition hover:bg-ink/10 hover:text-textMain"
                >
                  ×
                </button>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-4">
                {[
                  { key: 'zc', color: '#ea4335', desc: isZh ? '旋律、节奏、和声' : 'Melody, rhythm, harmony' },
                  { key: 'zs', color: '#188038', desc: isZh ? '文化、音色、语境' : 'Culture, timbre, context' },
                  { key: 'za', color: '#1a73e8', desc: isZh ? '效价、唤醒度' : 'Valence, arousal' }
                ].map((item) => (
                  <div key={item.key} className="rounded-xl border border-ink/15 bg-white p-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-3 w-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="font-mono text-xs font-bold text-textMain">{item.key}</span>
                    </div>
                    <p className="mt-1 text-[10px] text-textSub">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom hint */}
      <div className="mt-8 text-center">
        <p className="text-xs text-textSub">{hint}</p>
      </div>
    </SectionShell>
  )
}
