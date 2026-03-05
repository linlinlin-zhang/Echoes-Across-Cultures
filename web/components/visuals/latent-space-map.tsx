'use client'

import { useMemo } from 'react'

import { cultureNodes, otDemoRoutes, songPoints, type SongPoint } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { useSceneStore } from '@/components/state/scene-store'
import { clamp, cn } from '@/lib/utils'

type FactorKey = 'zc' | 'zs' | 'za'
type GalleryMode = 'observe' | 'disentangle' | 'transport'

type LatentSpaceMapProps = {
  galleryMode: GalleryMode
  cultureFilter: string
  activeFactors: Record<FactorKey, boolean>
  routeIndex: number
  energy: number
}

type Point = {
  song: SongPoint
  x: number
  y: number
}

const colorByFactor: Record<FactorKey, string> = {
  zc: '#ea4335',
  zs: '#188038',
  za: '#1a73e8'
}

const range = {
  width: 320,
  height: 210,
  extent: 1.95
}

function getFactorVector(song: SongPoint, factor: FactorKey): [number, number] {
  if (factor === 'zc') return [song.zcVector[0], song.zcVector[1]]
  if (factor === 'zs') return [song.zsVector[0], song.zsVector[1]]
  return [song.zaVector[0], song.zaVector[1]]
}

function toPoint(song: SongPoint, factor: FactorKey, mode: GalleryMode, energy: number): Point {
  const [vx, vy] = getFactorVector(song, factor)

  const modeScale = mode === 'observe' ? 0.7 : mode === 'disentangle' ? 0.98 : 0.82
  const factorScale = factor === 'za' ? 1.06 : factor === 'zs' ? 0.94 : 1
  const scale = modeScale * factorScale * (0.88 + energy * 0.22)

  const x = clamp(vx * scale, -range.extent, range.extent)
  const y = clamp(vy * scale, -range.extent, range.extent)

  return { song, x, y }
}

function project(x: number, y: number) {
  const px = ((x + range.extent) / (range.extent * 2)) * range.width
  const py = ((-y + range.extent) / (range.extent * 2)) * range.height
  return { x: px, y: py }
}

function computeCentroids(points: Point[]) {
  const bucket = new Map<string, { x: number; y: number; count: number }>()

  points.forEach((point) => {
    const prev = bucket.get(point.song.culture)
    if (!prev) {
      bucket.set(point.song.culture, { x: point.x, y: point.y, count: 1 })
      return
    }
    prev.x += point.x
    prev.y += point.y
    prev.count += 1
  })

  const centroids = new Map<string, { x: number; y: number }>()
  bucket.forEach((value, key) => {
    centroids.set(key, {
      x: value.x / value.count,
      y: value.y / value.count
    })
  })
  return centroids
}

export function LatentSpaceMap({ galleryMode, cultureFilter, activeFactors, routeIndex, energy }: LatentSpaceMapProps) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  const routeIdToCulture = useMemo(() => new Map(cultureNodes.map((node) => [node.id, node.name])), [])

  const routeCultures = useMemo(() => {
    const safe = ((routeIndex % otDemoRoutes.length) + otDemoRoutes.length) % otDemoRoutes.length
    const route = otDemoRoutes[safe] ?? otDemoRoutes[0]
    return route.map((id) => routeIdToCulture.get(id) ?? id)
  }, [routeIdToCulture, routeIndex])

  const routeSet = useMemo(() => new Set(routeCultures), [routeCultures])

  const visibleSongs = useMemo(() => {
    return cultureFilter === 'All' ? songPoints : songPoints.filter((song) => song.culture === cultureFilter)
  }, [cultureFilter])

  const panelPoints = useMemo(
    () => ({
      zc: visibleSongs.map((song) => toPoint(song, 'zc', galleryMode, energy)),
      zs: visibleSongs.map((song) => toPoint(song, 'zs', galleryMode, energy)),
      za: visibleSongs.map((song) => toPoint(song, 'za', galleryMode, energy))
    }),
    [energy, galleryMode, visibleSongs]
  )

  const zsCentroids = useMemo(() => computeCentroids(panelPoints.zs), [panelPoints.zs])

  const zsRouteLine = useMemo(() => {
    const points = routeCultures
      .map((culture) => {
        const centroid = zsCentroids.get(culture)
        if (!centroid) return null
        const projected = project(centroid.x, centroid.y)
        return `${projected.x},${projected.y}`
      })
      .filter((item): item is string => Boolean(item))

    return points.join(' ')
  }, [routeCultures, zsCentroids])

  const panelMeta: Array<{ key: FactorKey; titleZh: string; titleEn: string }> = [
    { key: 'zc', titleZh: '内容 zc', titleEn: 'Content zc' },
    { key: 'zs', titleZh: '文化 zs', titleEn: 'Culture zs' },
    { key: 'za', titleZh: '情感 za', titleEn: 'Affect za' }
  ]

  return (
    <div className="h-full overflow-auto p-3">
      <div className="grid gap-3 xl:grid-cols-3">
        {panelMeta.map((panel) => {
          const enabled = activeFactors[panel.key]
          const points = panelPoints[panel.key]
          const color = colorByFactor[panel.key]

          return (
            <div key={panel.key} className={cn('rounded-2xl border border-ink/15 bg-white/86 p-2 transition', enabled ? '' : 'opacity-35 grayscale')}>
              <p className="px-1 text-xs font-semibold text-textMain">{isZh ? panel.titleZh : panel.titleEn}</p>

              <svg viewBox={`0 0 ${range.width} ${range.height}`} className="mt-1.5 h-[180px] w-full rounded-xl border border-ink/15 bg-[#f8fbff]">
                {Array.from({ length: 4 }).map((_, i) => {
                  const x = ((i + 1) / 5) * range.width
                  const y = ((i + 1) / 5) * range.height
                  return (
                    <g key={`${panel.key}-grid-${i}`}>
                      <line x1={x} y1={0} x2={x} y2={range.height} stroke="rgba(70,80,100,0.14)" strokeWidth={1} />
                      <line x1={0} y1={y} x2={range.width} y2={y} stroke="rgba(70,80,100,0.14)" strokeWidth={1} />
                    </g>
                  )
                })}

                <line x1={range.width / 2} y1={0} x2={range.width / 2} y2={range.height} stroke="rgba(38,45,61,0.24)" strokeWidth={1} />
                <line x1={0} y1={range.height / 2} x2={range.width} y2={range.height / 2} stroke="rgba(38,45,61,0.24)" strokeWidth={1} />

                {panel.key === 'zs' && galleryMode === 'transport' && zsRouteLine ? <polyline points={zsRouteLine} fill="none" stroke="#188038" strokeWidth={2.2} strokeDasharray="6 4" opacity={0.86} /> : null}

                {points.map((point) => {
                  const projected = project(point.x, point.y)
                  const highlighted = selectedSongId === point.song.id || hoveredSongId === point.song.id
                  const routed = panel.key === 'zs' && routeSet.has(point.song.culture)

                  return (
                    <circle
                      key={`${panel.key}-${point.song.id}`}
                      cx={projected.x}
                      cy={projected.y}
                      r={highlighted ? 4.8 : routed ? 3.7 : 2.8}
                      fill={color}
                      fillOpacity={highlighted ? 1 : routed ? 0.92 : 0.72}
                      stroke={highlighted ? '#1f2637' : 'none'}
                      strokeWidth={highlighted ? 1.3 : 0}
                      onMouseEnter={() => setHoveredSongId(point.song.id)}
                      onMouseLeave={() => setHoveredSongId(null)}
                      onClick={() => setSelectedSongId(point.song.id)}
                    />
                  )
                })}
              </svg>
            </div>
          )
        })}
      </div>
    </div>
  )
}
