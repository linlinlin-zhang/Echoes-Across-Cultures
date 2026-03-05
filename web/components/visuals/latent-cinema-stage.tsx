'use client'

import { useEffect, useMemo, useState } from 'react'

import { cultureNodes, otDemoRoutes, songPoints, type SongPoint } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { clamp } from '@/lib/utils'

type GalleryMode = 'observe' | 'disentangle' | 'transport'
type FactorState = Record<'zc' | 'zs' | 'za', boolean>

type LatentCinemaStageProps = {
  galleryMode: GalleryMode
  cultureFilter: string
  factorState: FactorState
  routeIndex: number
  energy: number
  sceneStep: number
}

type FactorKey = 'zc' | 'zs' | 'za'

type Dot = {
  id: string
  song: SongPoint
  factor: FactorKey
  x: number
  y: number
}

const stage = {
  width: 1120,
  height: 620,
  padX: 66
}

const colorByFactor: Record<FactorKey, string> = {
  zc: '#ea4335',
  zs: '#188038',
  za: '#1a73e8'
}

const laneNameByFactor: Record<FactorKey, string> = {
  zc: 'CONTENT',
  zs: 'CULTURE',
  za: 'AFFECT'
}

function laneY(mode: GalleryMode, factor: FactorKey) {
  if (mode === 'observe') {
    return factor === 'zc' ? 246 : factor === 'zs' ? 312 : 380
  }
  if (mode === 'disentangle') {
    return factor === 'zc' ? 188 : factor === 'zs' ? 314 : 442
  }
  return factor === 'zc' ? 214 : factor === 'zs' ? 314 : 414
}

function factorVector(song: SongPoint, factor: FactorKey): [number, number] {
  if (factor === 'zc') return [song.zcVector[0], song.zcVector[1]]
  if (factor === 'zs') return [song.zsVector[0], song.zsVector[1]]
  return [song.zaVector[0], song.zaVector[1]]
}

function ribbonPath(baseY: number, phase: number, amp: number, freqScale: number, drift: number) {
  const points: string[] = []
  const steps = 18
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps
    const x = stage.padX + t * (stage.width - stage.padX * 2)
    const y = baseY + Math.sin(phase * 0.85 + i * freqScale + drift) * amp + Math.cos(phase * 0.45 + i * 0.38) * amp * 0.26
    points.push(`${x},${y}`)
  }
  return points.join(' ')
}

export function LatentCinemaStage({ galleryMode, cultureFilter, factorState, routeIndex, energy, sceneStep }: LatentCinemaStageProps) {
  const [phase, setPhase] = useState(0)

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const auditionFactor = useSceneStore((state) => state.auditionFactor)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      setPhase((prev) => (prev + 0.015 + energy * 0.02) % (Math.PI * 2 * 1000))
      raf = window.requestAnimationFrame(tick)
    }
    raf = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(raf)
  }, [energy])

  const filteredSongs = useMemo(() => (cultureFilter === 'All' ? songPoints : songPoints.filter((song) => song.culture === cultureFilter)), [cultureFilter])

  const routeIdToCulture = useMemo(() => new Map(cultureNodes.map((node) => [node.id, node.name])), [])
  const routeCultures = useMemo(() => {
    const safe = ((routeIndex % otDemoRoutes.length) + otDemoRoutes.length) % otDemoRoutes.length
    const route = otDemoRoutes[safe] ?? otDemoRoutes[0]
    return route.map((id) => routeIdToCulture.get(id) ?? id)
  }, [routeIdToCulture, routeIndex])
  const routeSet = useMemo(() => new Set(routeCultures), [routeCultures])

  const dots = useMemo(() => {
    const list: Dot[] = []

    ;(['zc', 'zs', 'za'] as const).forEach((factor, laneIndex) => {
      const baseY = laneY(galleryMode, factor)
      const laneSpread = galleryMode === 'disentangle' ? 82 : 54
      const lanePull = galleryMode === 'transport' && factor === 'zs' ? -26 : 0

      filteredSongs.forEach((song, index) => {
        const [vx, vy] = factorVector(song, factor)
        const t = filteredSongs.length <= 1 ? 0.5 : index / (filteredSongs.length - 1)
        const baseX = stage.padX + t * (stage.width - stage.padX * 2)

        const waveX = Math.sin(phase * 0.92 + index * 0.42 + laneIndex * 0.8) * (8 + energy * 11)
        const waveY = Math.cos(phase * 0.8 + index * 0.31 + laneIndex * 0.54) * (4 + energy * 8)

        const transferBias = galleryMode === 'transport' && routeSet.has(song.culture) ? (factor === 'zs' ? -26 : -10) : 0

        const x = baseX + vx * 46 + waveX
        const y = baseY + vy * laneSpread + waveY + lanePull + transferBias

        list.push({
          id: `${factor}-${song.id}`,
          song,
          factor,
          x,
          y
        })
      })
    })

    return list
  }, [energy, filteredSongs, galleryMode, phase, routeSet])

  const centroidByCulture = useMemo(() => {
    const bucket = new Map<string, { x: number; y: number; count: number }>()
    dots
      .filter((dot) => dot.factor === 'zs')
      .forEach((dot) => {
        const current = bucket.get(dot.song.culture)
        if (!current) {
          bucket.set(dot.song.culture, { x: dot.x, y: dot.y, count: 1 })
          return
        }
        current.x += dot.x
        current.y += dot.y
        current.count += 1
      })

    const out = new Map<string, { x: number; y: number }>()
    bucket.forEach((value, key) => {
      out.set(key, {
        x: value.x / value.count,
        y: value.y / value.count
      })
    })
    return out
  }, [dots])

  const transferPolyline = useMemo(() => {
    const points = routeCultures
      .map((culture) => {
        const centroid = centroidByCulture.get(culture)
        if (!centroid) return null
        return `${centroid.x},${centroid.y}`
      })
      .filter((value): value is string => Boolean(value))

    return points.join(' ')
  }, [centroidByCulture, routeCultures])

  const ribbons = useMemo(() => {
    const ampBase = 24 + energy * 18
    return {
      zc: ribbonPath(laneY(galleryMode, 'zc'), phase, ampBase * 0.7, 0.66, 0.3),
      zs: ribbonPath(laneY(galleryMode, 'zs'), phase, ampBase * 0.86, 0.62, 1.2),
      za: ribbonPath(laneY(galleryMode, 'za'), phase, ampBase * 0.72, 0.69, 2.1)
    }
  }, [energy, galleryMode, phase])

  const sceneFlash = useMemo(() => clamp(0.08 + (sceneStep % 2) * 0.05 + energy * 0.08, 0.05, 0.2), [energy, sceneStep])

  return (
    <div className="relative h-full w-full overflow-hidden bg-[radial-gradient(circle_at_20%_20%,rgba(26,115,232,0.08),transparent_35%),radial-gradient(circle_at_80%_70%,rgba(24,128,56,0.08),transparent_38%),linear-gradient(180deg,#fbfdff_0%,#f5f9ff_100%)]">
      <svg viewBox={`0 0 ${stage.width} ${stage.height}`} className="h-full w-full">
        <defs>
          <filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect x={0} y={0} width={stage.width} height={stage.height} fill={`rgba(26,115,232,${sceneFlash})`} />

        {(['zc', 'zs', 'za'] as const).map((factor) => {
          const emphasized = auditionFactor === factor
          return (
            <g key={`lane-${factor}`}>
              <polyline
                points={ribbons[factor]}
                fill="none"
                stroke={colorByFactor[factor]}
                strokeWidth={factorState[factor] ? (emphasized ? 4.4 : 2.4) : 1}
                strokeOpacity={factorState[factor] ? (emphasized ? 0.78 : 0.34) : 0.1}
                filter="url(#softGlow)"
              />
              <text x={26} y={laneY(galleryMode, factor) - 12} fontSize={11} fill={colorByFactor[factor]} opacity={factorState[factor] ? (emphasized ? 1 : 0.7) : 0.35}>
                {laneNameByFactor[factor]}
              </text>
            </g>
          )
        })}

        {galleryMode === 'transport' && transferPolyline ? (
          <polyline points={transferPolyline} fill="none" stroke="#188038" strokeWidth={4} strokeOpacity={0.48} strokeDasharray="14 8" filter="url(#softGlow)" />
        ) : null}

        {dots.map((dot) => {
          const enabled = factorState[dot.factor]
          if (!enabled) return null

          const highlighted = selectedSongId === dot.song.id || hoveredSongId === dot.song.id
          const routed = galleryMode === 'transport' && routeSet.has(dot.song.culture)
          const emphasized = auditionFactor === dot.factor

          return (
            <circle
              key={dot.id}
              cx={dot.x}
              cy={dot.y}
              r={highlighted ? 5.4 : emphasized ? 4.4 : routed ? 3.8 : 2.9}
              fill={colorByFactor[dot.factor]}
              fillOpacity={highlighted ? 1 : emphasized ? 0.96 : routed ? 0.9 : 0.7}
              stroke={highlighted ? '#0f172a' : emphasized ? '#ffffff' : 'none'}
              strokeWidth={highlighted ? 1.4 : emphasized ? 0.8 : 0}
              onMouseEnter={() => setHoveredSongId(dot.song.id)}
              onMouseLeave={() => setHoveredSongId(null)}
              onClick={() => setSelectedSongId(dot.song.id)}
              style={{ cursor: 'pointer' }}
            />
          )
        })}
      </svg>
    </div>
  )
}
