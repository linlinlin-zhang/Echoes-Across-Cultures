'use client'

import { useEffect, useMemo, useState } from 'react'

import { useSceneStore } from '@/components/state/scene-store'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cultureNodes, otDemoRoutes, songPoints } from '@/data/mock-data'
import { buildFactorMetrics } from '@/lib/factor-mapping'
import { clamp } from '@/lib/utils'

type GalleryMode = 'observe' | 'disentangle' | 'transport'
type FactorState = Record<'zc' | 'zs' | 'za', boolean>
type FactorKey = 'zc' | 'zs' | 'za'

type LatentCinemaStageProps = {
  galleryMode: GalleryMode
  cultureFilter: string
  factorState: FactorState
  routeIndex: number
  energy: number
  sceneStep: number
}

const stage = {
  width: 1120,
  height: 620
}

const factorColor: Record<FactorKey, string> = {
  zc: '#ea4335',
  zs: '#188038',
  za: '#1a73e8'
}

const lanes: Record<FactorKey, { y: number; h: number }> = {
  zc: { y: 78, h: 148 },
  zs: { y: 244, h: 148 },
  za: { y: 410, h: 148 }
}

export function LatentCinemaStage({ galleryMode, cultureFilter, factorState, routeIndex, energy, sceneStep }: LatentCinemaStageProps) {
  const [phase, setPhase] = useState(0)

  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const auditionFactor = useSceneStore((state) => state.auditionFactor)
  const auditionTrace = useSceneStore((state) => state.auditionTrace)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      setPhase((prev) => (prev + 0.012 + energy * 0.015) % (Math.PI * 4000))
      raf = window.requestAnimationFrame(tick)
    }
    raf = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(raf)
  }, [energy])

  const filteredSongs = useMemo(() => {
    const pool = cultureFilter === 'All' ? songPoints : songPoints.filter((song) => song.culture === cultureFilter)
    return pool.length > 0 ? pool : songPoints
  }, [cultureFilter])

  const focusSong = useMemo(() => filteredSongs.find((item) => item.id === (selectedSongId ?? hoveredSongId)) ?? filteredSongs[0], [filteredSongs, hoveredSongId, selectedSongId])

  const metrics = useMemo(() => buildFactorMetrics(focusSong), [focusSong])

  const routeIdToCulture = useMemo(() => new Map(cultureNodes.map((node) => [node.id, node.name])), [])
  const routeCultures = useMemo(() => {
    const safe = ((routeIndex % otDemoRoutes.length) + otDemoRoutes.length) % otDemoRoutes.length
    const route = otDemoRoutes[safe] ?? otDemoRoutes[0]
    return route.map((id) => routeIdToCulture.get(id) ?? id)
  }, [routeIdToCulture, routeIndex])

  const anchorSongs = useMemo(() => filteredSongs.slice(0, 12), [filteredSongs])

  const anchorX = (index: number, total: number) => {
    if (total <= 1) return stage.width / 2
    return 62 + (index / (total - 1)) * (stage.width - 124)
  }

  const transportProgress = (Math.sin(phase * 0.42 + sceneStep * 0.8) + 1) / 2

  const zcActive = factorState.zc
  const zsActive = factorState.zs
  const zaActive = factorState.za

  const playheadX = 298 + ((phase * 52) % 700)
  const emotionX = 482 + ((metrics.zaValence + 1) / 2) * 508
  const emotionY = lanes.za.y + 18 + (1 - (metrics.zaArousal + 1) / 2) * (lanes.za.h - 36)

  return (
    <div className="relative h-full w-full overflow-hidden bg-[radial-gradient(circle_at_15%_18%,rgba(26,115,232,0.07),transparent_38%),radial-gradient(circle_at_85%_82%,rgba(24,128,56,0.08),transparent_36%),linear-gradient(180deg,#fbfdff_0%,#f5f9ff_100%)]">
      <svg viewBox={`0 0 ${stage.width} ${stage.height}`} className="h-full w-full">
        <defs>
          <filter id="laneGlow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect x={0} y={0} width={stage.width} height={stage.height} fill={`rgba(26,115,232,${clamp(0.04 + energy * 0.04, 0.03, 0.12)})`} />

        <text x={24} y={34} fontSize={12} fill="#5f6b7a">{isZh ? '样本锚点（点击切换）' : 'sample anchors (click to switch)'}</text>
        {anchorSongs.map((song, index) => {
          const x = anchorX(index, anchorSongs.length)
          const active = song.id === focusSong.id
          return (
            <g key={song.id}>
              <circle
                cx={x}
                cy={46}
                r={active ? 7 : 4.5}
                fill={active ? '#0f172a' : '#94a3b8'}
                fillOpacity={active ? 0.9 : 0.52}
                stroke="#ffffff"
                strokeWidth={active ? 1.2 : 0.7}
                onMouseEnter={() => setHoveredSongId(song.id)}
                onMouseLeave={() => setHoveredSongId(null)}
                onClick={() => setSelectedSongId(song.id)}
                style={{ cursor: 'pointer' }}
              />
              {active ? <text x={x - 24} y={62} fontSize={10} fill="#334155">{song.title.replace('Echo Piece ', '#')}</text> : null}
            </g>
          )
        })}

        {(['zc', 'zs', 'za'] as const).map((factor) => {
          const lane = lanes[factor]
          const enabled = factorState[factor]
          const emphasized = auditionFactor === factor
          return (
            <g key={`lane-${factor}`} opacity={enabled ? 1 : 0.23}>
              <rect
                x={26}
                y={lane.y}
                width={stage.width - 52}
                height={lane.h}
                rx={22}
                fill="#ffffff"
                fillOpacity={0.82}
                stroke={factorColor[factor]}
                strokeOpacity={emphasized ? 0.5 : 0.18}
                strokeWidth={emphasized ? 2.2 : 1.1}
                filter={emphasized ? 'url(#laneGlow)' : undefined}
              />
              <text x={44} y={lane.y + 24} fontSize={12} fill={factorColor[factor]}>
                {factor === 'zc'
                  ? isZh
                    ? 'zc 内容 Content · 旋律/节奏'
                    : 'zc content · melody/rhythm'
                  : factor === 'zs'
                    ? isZh
                      ? 'zs 文化 Culture · 音色/乐器'
                      : 'zs culture · timbre/instrument'
                    : isZh
                      ? 'za 情感 Affect · Valence/Arousal'
                      : 'za affect · valence/arousal'}
              </text>
            </g>
          )
        })}

        {zcActive ? (
          <g>
            {Array.from({ length: 8 }).map((_, index) => {
              const x = 298 + index * 86
              const level = metrics.melodySteps[index] ?? 0.2
              const h = 18 + level * 86
              const y = lanes.zc.y + lanes.zc.h - 24 - h
              return <rect key={`zc-step-${index}`} x={x} y={y} width={58} height={h} rx={10} fill="#ea4335" fillOpacity={0.2 + level * 0.65} />
            })}
            <line x1={playheadX} y1={lanes.zc.y + 32} x2={playheadX} y2={lanes.zc.y + lanes.zc.h - 18} stroke="#ea4335" strokeOpacity={0.76} strokeWidth={2.2} />
            <text x={44} y={lanes.zc.y + 50} fontSize={11} fill="#5f6b7a">
              {isZh ? `密度 ${Math.round(metrics.rhythmDensity * 100)}% · 点击 zc 按键听节奏轮廓` : `density ${Math.round(metrics.rhythmDensity * 100)}% · hit zc pad to hear rhythm contour`}
            </text>
          </g>
        ) : null}

        {zsActive ? (
          <g>
            {metrics.timbreBars.map((bar, index) => {
              const x = 332 + index * 84
              const h = 16 + bar * 84
              const y = lanes.zs.y + lanes.zs.h - 20 - h
              return <rect key={`zs-bar-${index}`} x={x} y={y} width={48} height={h} rx={8} fill="#188038" fillOpacity={0.2 + bar * 0.58} />
            })}

            <text x={44} y={lanes.zs.y + 50} fontSize={11} fill="#5f6b7a">
              {isZh ? `文化 ${focusSong.culture} · 音色 ${metrics.cultureDescriptorZh}` : `culture ${focusSong.culture} · timbre ${metrics.cultureDescriptorEn}`}
            </text>

            {galleryMode === 'transport' ? (
              <g>
                {routeCultures.map((culture, index) => {
                  const x = 352 + (index / Math.max(1, routeCultures.length - 1)) * 442
                  return (
                    <g key={`route-node-${culture}-${index}`}>
                      <circle cx={x} cy={lanes.zs.y + 118} r={11} fill="#188038" fillOpacity={0.18} stroke="#188038" strokeOpacity={0.46} />
                      <text x={x - 32} y={lanes.zs.y + 138} fontSize={10} fill="#3f4d5e">{culture.slice(0, 10)}</text>
                    </g>
                  )
                })}

                {routeCultures.length > 1 ? (
                  <circle
                    cx={352 + transportProgress * 442}
                    cy={lanes.zs.y + 118}
                    r={9}
                    fill="#188038"
                    fillOpacity={0.92}
                    stroke="#ffffff"
                    strokeWidth={1.4}
                    filter="url(#laneGlow)"
                  />
                ) : null}
              </g>
            ) : null}
          </g>
        ) : null}

        {zaActive ? (
          <g>
            <rect x={482} y={lanes.za.y + 18} width={508} height={lanes.za.h - 36} rx={16} fill="#ffffff" fillOpacity={0.68} stroke="#1a73e8" strokeOpacity={0.28} />
            <line x1={736} y1={lanes.za.y + 18} x2={736} y2={lanes.za.y + lanes.za.h - 18} stroke="#1a73e8" strokeOpacity={0.18} />
            <line x1={482} y1={lanes.za.y + lanes.za.h / 2} x2={990} y2={lanes.za.y + lanes.za.h / 2} stroke="#1a73e8" strokeOpacity={0.18} />

            <circle cx={emotionX} cy={emotionY} r={12 + Math.sin(phase * 0.8) * 2.2} fill="#1a73e8" fillOpacity={0.88} stroke="#ffffff" strokeWidth={1.2} filter="url(#laneGlow)" />

            <text x={44} y={lanes.za.y + 50} fontSize={11} fill="#5f6b7a">
              {isZh ? `Valence ${metrics.zaValence.toFixed(2)} · Arousal ${metrics.zaArousal.toFixed(2)}（点位越高=越兴奋）` : `valence ${metrics.zaValence.toFixed(2)} · arousal ${metrics.zaArousal.toFixed(2)} (higher point = higher arousal)`}
            </text>
          </g>
        ) : null}

        <rect x={24} y={574} width={stage.width - 48} height={28} rx={12} fill="#ffffff" fillOpacity={0.84} stroke="#d0d8e5" />
        <text x={38} y={593} fontSize={11} fill="#4b5563">
          {auditionTrace ? (isZh ? `联动事件：${auditionTrace.summaryZh}` : `Linked event: ${auditionTrace.summaryEn}`) : isZh ? '联动事件：点击下方因子声音映射台任一按键' : 'Linked event: trigger any pad in the factor mapper below'}
        </text>
      </svg>
    </div>
  )
}
