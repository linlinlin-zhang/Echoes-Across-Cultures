'use client'

import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Canvas, ThreeEvent, useFrame } from '@react-three/fiber'
import { Line, OrbitControls, Stars } from '@react-three/drei'

import { cultureNodes, otDemoRoutes, songPoints, type SongPoint } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp } from '@/lib/utils'
import { useSceneStore } from '@/components/state/scene-store'

type FactorKey = 'zc' | 'zs' | 'za'

type GalleryMode = 'observe' | 'disentangle' | 'transport'

type LatentSpaceCanvasProps = {
  galleryMode: GalleryMode
  cultureFilter: string
  activeFactors: Record<FactorKey, boolean>
  routeIndex: number
  energy: number
}

type FactorPoint = {
  song: SongPoint
  factor: FactorKey
  base: THREE.Vector3
  separated: THREE.Vector3
  color: THREE.Color
}

const factorDirection: Record<FactorKey, THREE.Vector3> = {
  zc: new THREE.Vector3(1.55, -0.66, 0.42),
  zs: new THREE.Vector3(-1.42, 0.74, 1.08),
  za: new THREE.Vector3(0.18, 1.6, -1.34)
}

const factorColorHex: Record<FactorKey, string> = {
  zc: '#ea4335',
  zs: '#188038',
  za: '#1a73e8'
}

const factorColor = {
  zc: new THREE.Color(factorColorHex.zc),
  zs: new THREE.Color(factorColorHex.zs),
  za: new THREE.Color(factorColorHex.za)
}

const cultureIdToName = new Map(cultureNodes.map((node) => [node.id, node.name]))

function seededRandom(seed: number) {
  let value = seed >>> 0
  return () => {
    value = (1664525 * value + 1013904223) >>> 0
    return value / 4294967295
  }
}

function buildFactorPoints(songs: SongPoint[]) {
  const random = seededRandom(17)
  const noise = () => (random() * 2 - 1) * 0.58

  const points: FactorPoint[] = []
  songs.forEach((song) => {
    ;(['zc', 'zs', 'za'] as const).forEach((factor) => {
      const latent = factor === 'zc' ? song.zcVector : factor === 'zs' ? song.zsVector : [song.zaVector[0], song.zaVector[1], 0]
      const base = new THREE.Vector3(latent[0] * 0.4 + noise(), latent[1] * 0.4 + noise(), latent[2] * 0.4 + noise())
      const separated = base.clone().add(factorDirection[factor].clone().multiplyScalar(1.15))
      points.push({
        song,
        factor,
        base,
        separated,
        color: factorColor[factor].clone()
      })
    })
  })

  return points
}

function buildCultureAnchors() {
  const anchors = new Map<string, THREE.Vector3>()
  cultureNodes.forEach((node, index) => {
    const angle = (index / cultureNodes.length) * Math.PI * 2
    anchors.set(node.name, new THREE.Vector3(Math.cos(angle) * 3.1, Math.sin(angle * 1.7) * 0.92, Math.sin(angle) * 2.5))
  })
  return anchors
}

function computePointTarget(
  point: FactorPoint,
  mode: GalleryMode,
  separation: number,
  energy: number,
  routeCultures: Set<string>,
  cultureAnchors: Map<string, THREE.Vector3>,
  time: number,
  index: number
) {
  const target = point.base.clone().lerp(point.separated, clamp(0.18 + separation * 0.82, 0, 1))

  if (mode === 'disentangle') {
    target.lerp(point.separated, clamp(0.35 + energy * 0.42, 0, 1))
  }

  if (mode === 'transport') {
    const anchor = cultureAnchors.get(point.song.culture)
    if (anchor) {
      const routeBoost = routeCultures.has(point.song.culture) ? 0.24 + energy * 0.52 : 0.07 + energy * 0.2
      const wave = 0.52 + 0.48 * Math.sin(time * 1.35 + index * 0.17)
      target.lerp(anchor, clamp(routeBoost * wave, 0, 0.92))
    }
  }

  return target
}

function ParticleCloud({
  points,
  mode,
  separation,
  cultureFilter,
  activeFactors,
  routeCultures,
  cultureAnchors,
  energy
}: {
  points: FactorPoint[]
  mode: GalleryMode
  separation: number
  cultureFilter: string
  activeFactors: Record<FactorKey, boolean>
  routeCultures: string[]
  cultureAnchors: Map<string, THREE.Vector3>
  energy: number
}) {
  const instancedRef = useRef<THREE.InstancedMesh>(null)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  const routeSet = useMemo(() => new Set(routeCultures), [routeCultures])
  const positionBuffer = useRef(points.map((point) => point.base.clone()))

  const tempObject = useMemo(() => new THREE.Object3D(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])
  const neutralColor = useMemo(() => new THREE.Color('#c8cfdd'), [])
  const glowColor = useMemo(() => new THREE.Color('#ffffff'), [])

  useFrame((state, delta) => {
    const instance = instancedRef.current
    if (!instance) return

    const time = state.clock.getElapsedTime()
    const ease = clamp(0.1 + delta * 0.24, 0.06, 0.24)

    points.forEach((point, index) => {
      const target = computePointTarget(point, mode, separation, energy, routeSet, cultureAnchors, time, index)
      positionBuffer.current[index].lerp(target, ease)

      const factorEnabled = activeFactors[point.factor]
      const cultureMatch = cultureFilter === 'All' || point.song.culture === cultureFilter
      const visibility = factorEnabled ? (cultureMatch ? 1 : 0.24) : 0.04

      const baseScale = point.factor === 'za' ? 0.11 : 0.094
      const scale = Math.max(0.002, baseScale * visibility * (1 + energy * 0.16))

      tempObject.position.copy(positionBuffer.current[index])
      tempObject.scale.setScalar(scale)
      tempObject.updateMatrix()
      instance.setMatrixAt(index, tempObject.matrix)

      tempColor.copy(point.color)
      if (!factorEnabled) {
        tempColor.lerp(neutralColor, 0.84)
      } else if (!cultureMatch) {
        tempColor.lerp(neutralColor, 0.6)
      } else if (mode === 'transport' && routeSet.has(point.song.culture)) {
        tempColor.lerp(glowColor, 0.12 + 0.12 * (0.5 + 0.5 * Math.sin(time * 2 + index * 0.11)))
      }

      instance.setColorAt(index, tempColor)
    })

    instance.instanceMatrix.needsUpdate = true
    if (instance.instanceColor) {
      instance.instanceColor.needsUpdate = true
    }
  })

  const handleHover = (event: ThreeEvent<PointerEvent>) => {
    if (event.instanceId == null) return
    const point = points[event.instanceId]
    if (!point) return
    if (!activeFactors[point.factor]) return
    if (cultureFilter !== 'All' && point.song.culture !== cultureFilter) return
    setHoveredSongId(point.song.id)
  }

  const handleSelect = (event: ThreeEvent<MouseEvent>) => {
    if (event.instanceId == null) return
    const point = points[event.instanceId]
    if (!point) return
    if (!activeFactors[point.factor]) return
    if (cultureFilter !== 'All' && point.song.culture !== cultureFilter) return
    setSelectedSongId(point.song.id)
  }

  return (
    <instancedMesh
      ref={instancedRef}
      args={[undefined, undefined, points.length]}
      onPointerMove={handleHover}
      onPointerOut={() => setHoveredSongId(null)}
      onClick={handleSelect}
    >
      <sphereGeometry args={[0.11, 14, 14]} />
      <meshStandardMaterial transparent opacity={0.92} emissive={new THREE.Color('#f3f6ff')} emissiveIntensity={0.14} metalness={0.06} roughness={0.28} />
    </instancedMesh>
  )
}

function RouteLines({ routeCultures, cultureAnchors, energy }: { routeCultures: string[]; cultureAnchors: Map<string, THREE.Vector3>; energy: number }) {
  const segments = useMemo(() => {
    const lines: Array<{ id: string; points: [THREE.Vector3, THREE.Vector3, THREE.Vector3] }> = []
    for (let index = 0; index < routeCultures.length - 1; index += 1) {
      const source = cultureAnchors.get(routeCultures[index])
      const target = cultureAnchors.get(routeCultures[index + 1])
      if (!source || !target) continue

      const middle = source
        .clone()
        .add(target)
        .multiplyScalar(0.5)
      middle.y += 0.56 + index * 0.16

      lines.push({
        id: `${routeCultures[index]}-${routeCultures[index + 1]}`,
        points: [source.clone(), middle, target.clone()]
      })
    }
    return lines
  }, [cultureAnchors, routeCultures])

  return (
    <>
      {segments.map((segment, index) => (
        <Line key={segment.id} points={segment.points} color="#188038" transparent opacity={0.38 + energy * 0.42} lineWidth={1.8 + index * 0.15} />
      ))}
    </>
  )
}

function FactorGuides({ separation }: { separation: number }) {
  const haloRef = useRef<THREE.Group>(null)

  useFrame((_state, delta) => {
    if (!haloRef.current) return
    haloRef.current.rotation.y += delta * 0.08
    haloRef.current.rotation.x = Math.sin(_state.clock.elapsedTime * 0.2) * 0.04
  })

  const centers = useMemo(
    () =>
      (['zc', 'zs', 'za'] as const).map((factor) => ({
        factor,
        center: factorDirection[factor].clone().multiplyScalar(0.4 + separation * 0.74)
      })),
    [separation]
  )

  return (
    <group ref={haloRef}>
      {(Object.keys(factorDirection) as FactorKey[]).map((factor) => (
        <Line key={`axis-${factor}`} points={[new THREE.Vector3(0, 0, 0), factorDirection[factor].clone().multiplyScalar(3)]} color={factorColorHex[factor]} transparent opacity={0.32} lineWidth={1.1} />
      ))}

      {centers.map((item, index) => (
        <mesh key={`halo-${item.factor}`} position={[item.center.x, item.center.y, item.center.z]} rotation={[Math.PI / 2 + index * 0.35, 0, index * 0.28]}>
          <torusGeometry args={[0.44 + separation * 0.22, 0.022, 14, 42]} />
          <meshBasicMaterial color={factorColorHex[item.factor]} transparent opacity={0.24} />
        </mesh>
      ))}
    </group>
  )
}

function SelectionBeacon({
  points,
  mode,
  separation,
  cultureFilter,
  routeCultures,
  cultureAnchors,
  energy
}: {
  points: FactorPoint[]
  mode: GalleryMode
  separation: number
  cultureFilter: string
  routeCultures: string[]
  cultureAnchors: Map<string, THREE.Vector3>
  energy: number
}) {
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const beaconRef = useRef<THREE.Mesh>(null)
  const routeSet = useMemo(() => new Set(routeCultures), [routeCultures])
  const selectedPoint = useMemo(() => points.find((item) => item.song.id === selectedSongId && item.factor === 'zc') ?? null, [points, selectedSongId])
  const current = useRef(new THREE.Vector3(0, 0, 0))

  useFrame((state, delta) => {
    const beacon = beaconRef.current
    if (!beacon) return

    if (!selectedPoint) {
      beacon.visible = false
      return
    }

    if (cultureFilter !== 'All' && selectedPoint.song.culture !== cultureFilter) {
      beacon.visible = false
      return
    }

    const target = computePointTarget(selectedPoint, mode, separation, energy, routeSet, cultureAnchors, state.clock.getElapsedTime(), 0)
    current.current.lerp(target, clamp(0.08 + delta * 0.26, 0.08, 0.22))

    beacon.visible = true
    beacon.position.copy(current.current)
    beacon.rotation.z += delta * 1.1
    beacon.rotation.y += delta * 0.8
  })

  return (
    <mesh ref={beaconRef} visible={false}>
      <torusGeometry args={[0.19, 0.028, 16, 48]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.85} />
    </mesh>
  )
}

function SceneCore({ galleryMode, cultureFilter, activeFactors, routeIndex, energy }: LatentSpaceCanvasProps) {
  const separation = useSceneStore((state) => state.separation)
  const { reduceMotion } = useAccessibility()

  const points = useMemo(() => buildFactorPoints(songPoints), [])
  const cultureAnchors = useMemo(() => buildCultureAnchors(), [])

  const routeCultures = useMemo(() => {
    const safeIndex = ((routeIndex % otDemoRoutes.length) + otDemoRoutes.length) % otDemoRoutes.length
    const routeIds = otDemoRoutes[safeIndex] ?? otDemoRoutes[0]
    return routeIds.map((id) => cultureIdToName.get(id) ?? id)
  }, [routeIndex])

  const autoRotateSpeed = reduceMotion ? 0 : 0.16 + energy * 0.34

  return (
    <>
      <color attach="background" args={['#f6f8fc']} />
      <fog attach="fog" args={['#edf2fb', 4.2, 13.6]} />
      <ambientLight intensity={1.15} />
      <pointLight position={[4, 3, 2]} intensity={1.6} color={'#ea4335'} />
      <pointLight position={[-5, -2.2, 3]} intensity={1.45} color={'#188038'} />
      <pointLight position={[0.4, 3.4, -2.2]} intensity={1.35} color={'#1a73e8'} />
      <hemisphereLight intensity={0.45} groundColor={'#e4ebfb'} color={'#fdfefe'} />
      <Stars radius={30} depth={16} count={780} factor={3} saturation={0} fade speed={0.6} />

      <group position={[0, -0.08, 0]}>
        <FactorGuides separation={separation} />
        <RouteLines routeCultures={routeCultures} cultureAnchors={cultureAnchors} energy={energy} />

        <ParticleCloud
          points={points}
          mode={galleryMode}
          separation={separation}
          cultureFilter={cultureFilter}
          activeFactors={activeFactors}
          routeCultures={routeCultures}
          cultureAnchors={cultureAnchors}
          energy={energy}
        />

        <SelectionBeacon
          points={points}
          mode={galleryMode}
          separation={separation}
          cultureFilter={cultureFilter}
          routeCultures={routeCultures}
          cultureAnchors={cultureAnchors}
          energy={energy}
        />
      </group>

      <OrbitControls enablePan={false} enableZoom minDistance={2.3} maxDistance={8.2} autoRotate={!reduceMotion} autoRotateSpeed={autoRotateSpeed} />
    </>
  )
}

export function LatentSpaceCanvas(props: LatentSpaceCanvasProps) {
  return (
    <Canvas camera={{ position: [0.1, 0.78, 5.4], fov: 50 }} dpr={[1, 1.8]} gl={{ antialias: true, alpha: true }}>
      <SceneCore {...props} />
    </Canvas>
  )
}
