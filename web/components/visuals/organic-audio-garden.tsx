'use client'

import { useEffect, useRef, useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSceneStore } from '@/components/state/scene-store'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { songPoints } from '@/data/mock-data'
import { buildFactorMetrics } from '@/lib/factor-mapping'
import { Wind, Waves, Flame } from 'lucide-react'

// Organic visualization modes
 type GardenMode = 'wind' | 'water' | 'fire'

 interface GardenCell {
  id: number
  x: number
  y: number
  baseHeight: number
  currentHeight: number
  targetHeight: number
  color: string
  phase: number
}

 function generateOrganicGrid(rows: number, cols: number): GardenCell[] {
  const cells: GardenCell[] = []
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const noise = Math.sin(row * 0.5) * Math.cos(col * 0.5) * 0.3 + Math.random() * 0.2
      cells.push({
        id: row * cols + col,
        x: col,
        y: row,
        baseHeight: 0.3 + noise * 0.2,
        currentHeight: 0.3,
        targetHeight: 0.3,
        color: '#4a90d9',
        phase: Math.random() * Math.PI * 2
      })
    }
  }
  return cells
}

 // Wind mode - gentle flowing waves
 function WindVisualization({
  cells,
  intensity,
  factor
}: {
  cells: GardenCell[]
  intensity: number
  factor: 'zc' | 'zs' | 'za' | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [time, setTime] = useState(0)
  const animationRef = useRef<number>()

  useEffect(() => {
    const animate = () => {
      setTime(t => t + 0.02)
      animationRef.current = requestAnimationFrame(animate)
    }
    animate()
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [])

  const getCellStyle = (cell: GardenCell) => {
    const waveX = Math.sin(time + cell.x * 0.3) * 0.5
    const waveY = Math.cos(time * 0.7 + cell.y * 0.3) * 0.5
    const height = cell.baseHeight + (waveX + waveY) * 0.3 * intensity

    const colorMap = {
      zc: '#e85d4e',
      zs: '#2d8a5e',
      za: '#4a90d9',
      null: '#6b7280'
    }

    return {
      height: `${Math.max(10, height * 100)}%`,
      backgroundColor: colorMap[factor || 'null'],
      opacity: 0.4 + height * 0.6,
      transform: `scaleY(${0.8 + height * 0.4})`
    }
  }

  return (
    <div
      ref={containerRef}
      className="grid h-full w-full gap-0.5"
      style={{
        gridTemplateColumns: 'repeat(20, 1fr)',
        gridTemplateRows: 'repeat(8, 1fr)'
      }}
    >
      {cells.map(cell => (
        <motion.div
          key={cell.id}
          className="rounded-full transition-all duration-100"
          style={getCellStyle(cell)}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ duration: 0.5, delay: cell.id * 0.001 }}
        />
      ))}
    </div>
  )
}

 // Water mode - rippling circles
 function WaterVisualization({
  intensity,
  factor
}: {
  intensity: number
  factor: 'zc' | 'zs' | 'za' | null
}) {
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number; time: number }>>([])
  const rippleIdRef = useRef(0)

  useEffect(() => {
    if (!factor) return

    const interval = setInterval(() => {
      const newRipple = {
        id: ++rippleIdRef.current,
        x: 20 + Math.random() * 60,
        y: 20 + Math.random() * 60,
        time: Date.now()
      }
      setRipples(prev => [...prev.slice(-5), newRipple])
    }, 800 / (intensity + 0.5))

    return () => clearInterval(interval)
  }, [factor, intensity])

  const color = factor === 'zc' ? '#e85d4e' : factor === 'zs' ? '#2d8a5e' : '#4a90d9'

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* Background gradient */}
      <div
        className="absolute inset-0 opacity-30"
        style={{
          background: `radial-gradient(ellipse at center, ${color}22 0%, transparent 70%)`
        }}
      />

      {/* Ripples */}
      <AnimatePresence>
        {ripples.map(ripple => (
          <motion.div
            key={ripple.id}
            className="absolute rounded-full border-2"
            style={{
              left: `${ripple.x}%`,
              top: `${ripple.y}%`,
              borderColor: color,
              transform: 'translate(-50%, -50%)'
            }}
            initial={{ width: 0, height: 0, opacity: 1 }}
            animate={{ width: 300, height: 300, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 2, ease: 'easeOut' }}
          />
        ))}
      </AnimatePresence>

      {/* Floating orbs */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full blur-md"
          style={{
            backgroundColor: color,
            width: 20 + i * 10,
            height: 20 + i * 10
          }}
          animate={{
            x: [Math.random() * 100, Math.random() * 100 + 50, Math.random() * 100],
            y: [Math.random() * 50, Math.random() * 50 + 30, Math.random() * 50],
            opacity: [0.2, 0.5, 0.2]
          }}
          transition={{
            duration: 5 + i,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      ))}
    </div>
  )
}

 // Fire mode - dancing flames
 function FireVisualization({
  intensity,
  factor
}: {
  intensity: number
  factor: 'zc' | 'zs' | 'za' | null
}) {
  const particlesRef = useRef<HTMLDivElement>(null)

  const color = factor === 'zc' ? '#e85d4e' : factor === 'zs' ? '#2d8a5e' : '#4a90d9'

  return (
    <div ref={particlesRef} className="relative h-full w-full overflow-hidden">
      {/* Rising particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            backgroundColor: color,
            width: 4 + Math.random() * 8,
            height: 4 + Math.random() * 8,
            left: `${10 + Math.random() * 80}%`,
            bottom: 0,
            filter: 'blur(2px)'
          }}
          animate={{
            y: [-20, -200 - Math.random() * 100],
            x: [0, (Math.random() - 0.5) * 60, (Math.random() - 0.5) * 40],
            opacity: [0, 0.8, 0],
            scale: [0.5, 1.2, 0.3]
          }}
          transition={{
            duration: 2 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
            ease: 'easeOut'
          }}
        />
      ))}

      {/* Base glow */}
      <motion.div
        className="absolute bottom-0 left-1/2 h-32 w-full -translate-x-1/2 rounded-full blur-3xl"
        style={{ backgroundColor: color }}
        animate={{
          opacity: [0.3, 0.6 * intensity, 0.3],
          scaleX: [0.8, 1.2, 0.8]
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: 'easeInOut'
        }}
      />
    </div>
  )
}

 // Main component
 export function OrganicAudioGarden({ className }: { className?: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const [mode, setMode] = useState<GardenMode>('wind')
  const cells = useMemo(() => generateOrganicGrid(8, 20), [])

  const auditionFactor = useSceneStore(state => state.auditionFactor)
  const hoveredSongId = useSceneStore(state => state.hoveredSongId)
  const selectedSongId = useSceneStore(state => state.selectedSongId)

  const activeSong = useMemo(() => {
    return songPoints.find(s => s.id === (selectedSongId ?? hoveredSongId)) ?? songPoints[0]
  }, [hoveredSongId, selectedSongId])

  const metrics = useMemo(() => buildFactorMetrics(activeSong), [activeSong])

  // Calculate intensity based on factor activity
  const intensity = useMemo(() => {
    if (!auditionFactor) return 0.3
    const factorMetrics = {
      zc: metrics.zcStrength,
      zs: metrics.zsStrength,
      za: (metrics.zaArousal + 1) / 2
    }
    return 0.5 + factorMetrics[auditionFactor] * 0.5
  }, [auditionFactor, metrics])

  return (
    <div className={`flex flex-col overflow-hidden rounded-3xl border border-ink/15 bg-white backdrop-blur-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-ink/15 px-3 py-2">
        <div className="flex items-center gap-3">
          <span className="rounded-full bg-ink/10 px-3 py-1 text-xs font-medium text-textMain">
            {isZh ? '有机音景' : 'Organic Soundscape'}
          </span>
          <span className="text-xs text-textSub">
            {activeSong.title} · {activeSong.culture}
          </span>
        </div>

        {/* Mode switcher */}
        <div className="flex gap-1 rounded-full bg-ink/5 p-1">
          {[
            { key: 'wind', icon: Wind, label: isZh ? '风' : 'Wind' },
            { key: 'water', icon: Waves, label: isZh ? '水' : 'Water' },
            { key: 'fire', icon: Flame, label: isZh ? '火' : 'Fire' }
          ].map(({ key, icon: Icon, label }) => (
            <button
              key={key}
              onClick={() => setMode(key as GardenMode)}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition-all ${
                mode === key
                  ? 'bg-white text-za shadow-sm'
                  : 'text-textSub hover:text-textMain'
              }`}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Visualization area */}
      <div className="relative flex-1 p-2 min-h-0">
        {mode === 'wind' && <WindVisualization cells={cells} intensity={intensity} factor={auditionFactor} />}
        {mode === 'water' && <WaterVisualization intensity={intensity} factor={auditionFactor} />}
        {mode === 'fire' && <FireVisualization intensity={intensity} factor={auditionFactor} />}

        {/* Factor indicator overlay */}
        <AnimatePresence>
          {auditionFactor && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute left-4 top-4 rounded-full border border-ink/20 bg-white/90 px-3 py-1.5 shadow-sm backdrop-blur-sm"
            >
              <span
                className="text-xs font-semibold"
                style={{
                  color:
                    auditionFactor === 'zc'
                      ? '#e85d4e'
                      : auditionFactor === 'zs'
                      ? '#2d8a5e'
                      : '#4a90d9'
                }}
              >
                {auditionFactor === 'zc'
                  ? isZh
                    ? 'zc 内容脉动中'
                    : 'zc Content Flowing'
                  : auditionFactor === 'zs'
                  ? isZh
                    ? 'zs 文化共鸣中'
                    : 'zs Culture Resonating'
                  : isZh
                  ? 'za 情感涌动中'
                  : 'za Affect Surging'}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-px border-t border-ink/15 bg-ink/5">
        {[
          {
            key: 'zc',
            label: isZh ? '内容密度' : 'Content',
            value: `${Math.round(metrics.zcStrength * 100)}%`,
            color: '#e85d4e'
          },
          {
            key: 'zs',
            label: isZh ? '文化强度' : 'Culture',
            value: `${Math.round(metrics.zsStrength * 100)}%`,
            color: '#2d8a5e'
          },
          {
            key: 'za',
            label: isZh ? '情感唤醒' : 'Arousal',
            value: `${Math.round(((metrics.zaArousal + 1) / 2) * 100)}%`,
            color: '#4a90d9'
          }
        ].map(metric => (
          <div key={metric.key} className="px-4 py-3 text-center">
            <p className="text-[10px] uppercase tracking-wider text-textSub">{metric.label}</p>
            <p className="font-mono text-lg font-semibold" style={{ color: metric.color }}>
              {metric.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}


