'use client'

import { useState, useMemo, useRef } from 'react'
import { motion } from 'framer-motion'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { useSceneStore } from '@/components/state/scene-store'
import { songPoints } from '@/data/mock-data'
import { buildFactorMetrics } from '@/lib/factor-mapping'
import { ChevronRight, Sparkles, RotateCcw, Eye, Ear } from 'lucide-react'

// Mini 2D scatter plot showing latent space distribution
function LatentScatter({ songs, selectedId, hoveredId, onSelect }: {
  songs: typeof songPoints
  selectedId: string | null
  hoveredId: string | null
  onSelect: (id: string) => void
}) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  // Project 3D vectors to 2D for visualization
  const projectedPoints = useMemo(() => {
    return songs.map(song => {
      // Use zc and za for 2D projection
      const x = (song.zcVector[0] + 1.2) / 2.4 // Normalize to 0-1
      const y = (song.zaVector[1] + 1) / 2 // Normalize to 0-1
      return { song, x, y }
    })
  }, [songs])

  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl border border-ink/15 bg-white">
      {/* Grid lines */}
      <svg className="absolute inset-0 h-full w-full">
        {[0.25, 0.5, 0.75].map(pos => (
          <g key={pos}>
            <line x1={`${pos * 100}%`} y1="0" x2={`${pos * 100}%`} y2="100%" stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
            <line x1="0" y1={`${pos * 100}%`} x2="100%" y2={`${pos * 100}%`} stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
          </g>
        ))}

        {/* Axis labels */}
        <text x="50%" y="98%" fill="rgba(255,255,255,0.4)" fontSize={8} textAnchor="middle">{isZh ? 'zc 内容 ->' : 'zc Content ->'}</text>
        <text x="2%" y="50%" fill="rgba(255,255,255,0.4)" fontSize={8} textAnchor="start" transform="rotate(-90, 10, 50)">{isZh ? 'za 情感 ->' : 'za Affect ->'}</text>
      </svg>

      {/* Points */}
      {projectedPoints.map(({ song, x, y }) => {
        const isSelected = selectedId === song.id
        const isHovered = hoveredId === song.id
        // Color based on dominant culture
        const cultureColors: Record<string, string> = {
          'Western Pop': '#e85d4e',
          'Indian Classical': '#f9ab00',
          'Turkish Makam': '#8b5cf6',
          'Gamelan': '#2d8a5e',
          'West African Drumming': '#c73e2e',
          'Andean Folk': '#4a90d9',
          'Guqin': '#10b981',
          'Arabic Maqam': '#f59e0b'
        }
        const color = cultureColors[song.culture] || '#6b7280'

        return (
          <motion.button
            key={song.id}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${x * 100}%`, top: `${(1 - y) * 100}%` }}
            onClick={() => onSelect(song.id)}
            whileHover={{ scale: 1.5 }}
            animate={{
              scale: isSelected ? 1.8 : isHovered ? 1.4 : 1,
              opacity: isSelected ? 1 : isHovered ? 0.9 : 0.6
            }}
          >
            <div
              className="h-2 w-2 rounded-full"
              style={{
                backgroundColor: color,
                boxShadow: isSelected ? `0 0 10px ${color}` : 'none'
              }}
            />
            {isSelected && (
              <div className="absolute left-1/2 top-3 z-10 -translate-x-1/2 whitespace-nowrap rounded-full border border-ink/20 bg-white px-2 py-0.5 text-[8px] text-textMain shadow-sm">
                {song.title}
              </div>
            )}
          </motion.button>
        )
      })}
    </div>
  )
}

// Factor breakdown card with organic visualization
function FactorBreakdown({ song }: { song: typeof songPoints[0] }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const metrics = useMemo(() => buildFactorMetrics(song), [song])

  const factors = [
    {
      key: 'zc',
      name: isZh ? '内容' : 'Content',
      fullName: isZh ? '旋律与节奏' : 'Melody & Rhythm',
      value: metrics.zcStrength,
      color: '#e85d4e',
      icon: '🎵',
      details: [
        isZh ? `密度 ${Math.round(metrics.rhythmDensity * 100)}%` : `${Math.round(metrics.rhythmDensity * 100)}%`,
      ]
    },
    {
      key: 'zs',
      name: isZh ? '文化' : 'Culture',
      fullName: song.culture.length > 8 ? song.culture.slice(0, 8) + '...' : song.culture,
      value: metrics.zsStrength,
      color: '#2d8a5e',
      icon: '🌍',
      details: [
        isZh ? metrics.cultureDescriptorZh.slice(0, 6) : metrics.cultureDescriptorEn.slice(0, 10),
      ]
    },
    {
      key: 'za',
      name: isZh ? '情感' : 'Affect',
      fullName: isZh ? '效价与唤醒' : 'V & A',
      value: (metrics.zaArousal + 1) / 2,
      color: '#4a90d9',
      icon: '💫',
      details: [
        `${metrics.zaValence.toFixed(1)}/${metrics.zaArousal.toFixed(1)}`,
      ]
    }
  ]

  return (
    <div className="space-y-2">
      {factors.map((factor, index) => (
        <motion.div
          key={factor.key}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="group relative overflow-hidden rounded-xl border border-ink/15 bg-ink/5 p-2 transition-all hover:bg-ink/10"
        >
          {/* Progress bar background */}
          <div
            className="absolute bottom-0 left-0 top-0 opacity-10 transition-all group-hover:opacity-20"
            style={{ width: `${factor.value * 100}%`, backgroundColor: factor.color }}
          />

          <div className="relative flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-base">{factor.icon}</span>
              <div>
                <p className="text-xs font-semibold text-textMain leading-tight">{factor.name}</p>
                <p className="text-[9px] text-textSub truncate max-w-[100px]">{factor.fullName}</p>
              </div>
            </div>

            <div className="text-right">
              <p className="font-mono text-sm font-bold" style={{ color: factor.color }}>
                {Math.round(factor.value * 100)}%
              </p>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )
}

// Interactive culture selector
function CultureRibbon({ selected, onSelect }: { selected: string; onSelect: (c: string) => void }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const cultures = useMemo(() => ['All', ...Array.from(new Set(songPoints.map(s => s.culture)))], [])

  const scrollRef = useRef<HTMLDivElement>(null)

  return (
    <div className="relative">
      <div
        ref={scrollRef}
        className="scrollbar-hide flex gap-2 overflow-x-auto pb-2"
      >
        {cultures.map(culture => (
          <button
            key={culture}
            onClick={() => onSelect(culture)}
            className={`flex-shrink-0 rounded-full border px-3 py-1.5 text-xs transition-all ${
              selected === culture
                ? 'border-za/40 bg-za/10 text-za'
                : 'border-ink/15 bg-ink/5 text-textSub hover:border-ink/20 hover:bg-ink/10'
            }`}
          >
            {culture === 'All' ? (isZh ? '全部' : 'All') : culture}
          </button>
        ))}
      </div>

      {/* Fade edges */}
      <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-white to-transparent" />
    </div>
  )
}

// Main explorer component
export function LatentExplorer({ className }: { className?: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const selectedSongId = useSceneStore(state => state.selectedSongId)
  const hoveredSongId = useSceneStore(state => state.hoveredSongId)
  const setSelectedSongId = useSceneStore(state => state.setSelectedSongId)

  const [cultureFilter, setCultureFilter] = useState('All')
  const [viewMode, setViewMode] = useState<'explore' | 'analyze'>('explore')

  const filteredSongs = useMemo(() => {
    if (cultureFilter === 'All') return songPoints.slice(0, 24)
    return songPoints.filter(s => s.culture === cultureFilter).slice(0, 24)
  }, [cultureFilter])

  const activeSong = useMemo(() => {
    return songPoints.find(s => s.id === (selectedSongId ?? hoveredSongId)) ?? songPoints[0]
  }, [selectedSongId, hoveredSongId])

  return (
    <div className={`flex flex-col overflow-hidden rounded-3xl border border-ink/15 bg-white backdrop-blur-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-ink/15 px-4 py-3">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-amber-400" />
          <span className="text-xs font-semibold text-textMain">
            {isZh ? '潜空间探索器' : 'Latent Explorer'}
          </span>
        </div>

        <div className="flex gap-1 rounded-full bg-ink/5 p-1">
          <button
            onClick={() => setViewMode('explore')}
            className={`flex items-center gap-1 rounded-full px-3 py-1 text-xs transition ${
              viewMode === 'explore' ? 'bg-white text-za shadow-sm' : 'text-textSub hover:text-textMain'
            }`}
          >
            <Eye size={10} />
            {isZh ? '探索' : 'Explore'}
          </button>
          <button
            onClick={() => setViewMode('analyze')}
            className={`flex items-center gap-1 rounded-full px-3 py-1 text-xs transition ${
              viewMode === 'analyze' ? 'bg-white text-za shadow-sm' : 'text-textSub hover:text-textMain'
            }`}
          >
            <Ear size={10} />
            {isZh ? '分析' : 'Analyze'}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-3 overflow-auto">
        {/* Culture filter */}
        <div className="mb-2">
          <CultureRibbon selected={cultureFilter} onSelect={setCultureFilter} />
        </div>

        {viewMode === 'explore' ? (
          <div className="space-y-3">
            {/* Scatter plot */}
            <div className="h-32">
              <LatentScatter
                songs={filteredSongs}
                selectedId={selectedSongId}
                hoveredId={hoveredSongId}
                onSelect={setSelectedSongId}
              />
            </div>

            {/* Selected song info */}
            <div className="rounded-xl border border-ink/15 bg-ink/5 p-3">
              <p className="text-[10px] uppercase tracking-wider text-textSub">{isZh ? '当前样本' : 'Current Sample'}</p>
              <p className="mt-1 font-display text-base font-semibold text-textMain truncate">{activeSong.title}</p>
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="rounded-full bg-ink/10 px-2 py-0.5 text-[10px] text-textSub">{activeSong.culture}</span>
                <span className="rounded-full bg-ink/10 px-2 py-0.5 text-[10px] text-textSub">{activeSong.emotion}</span>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setSelectedSongId(null)}
                className="flex flex-1 items-center justify-center gap-1 rounded-xl border border-ink/15 bg-ink/5 py-2 text-xs text-textSub transition hover:bg-ink/10"
              >
                <RotateCcw size={12} />
                {isZh ? '重置' : 'Reset'}
              </button>
              <button className="flex flex-1 items-center justify-center gap-1 rounded-xl bg-za/10 py-2 text-xs font-semibold text-za transition hover:bg-za/20">
                {isZh ? '查看详情' : 'Details'}
                <ChevronRight size={12} />
              </button>
            </div>
          </div>
        ) : (
          <FactorBreakdown song={activeSong} />
        )}
      </div>

      {/* Stats footer */}
      <div className="grid grid-cols-3 gap-px border-t border-ink/15 bg-ink/5">
        {[
          { label: isZh ? '样本' : 'Samples', value: filteredSongs.length },
          { label: isZh ? '文化' : 'Cultures', value: new Set(filteredSongs.map(s => s.culture)).size },
          { label: isZh ? '维度' : 'Dimensions', value: '3D' }
        ].map(stat => (
          <div key={stat.label} className="px-3 py-2 text-center">
            <p className="text-[10px] text-textSub">{stat.label}</p>
            <p className="font-mono text-sm font-semibold text-textMain">{stat.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}


