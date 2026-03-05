'use client'

import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Canvas, ThreeEvent, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'

import { songPoints, type SongPoint } from '@/data/mock-data'
import { clamp } from '@/lib/utils'
import { useSceneStore } from '@/components/state/scene-store'

type FactorPoint = {
  song: SongPoint
  factor: 'zc' | 'zs' | 'za'
  base: THREE.Vector3
  target: THREE.Vector3
  color: THREE.Color
}

const factorColor = {
  zc: new THREE.Color('#ff6b6b'),
  zs: new THREE.Color('#4ecdc4'),
  za: new THREE.Color('#a55eea')
}

function seededRandom(seed: number) {
  let value = seed >>> 0
  return () => {
    value = (1664525 * value + 1013904223) >>> 0
    return value / 4294967295
  }
}

function buildFactorPoints(songs: SongPoint[]) {
  const random = seededRandom(7)
  const noise = () => (random() * 2 - 1) * 0.6
  const direction = {
    zc: new THREE.Vector3(1.6, -0.8, 0.45),
    zs: new THREE.Vector3(-1.4, 0.5, 1.2),
    za: new THREE.Vector3(0.25, 1.65, -1.35)
  }

  const points: FactorPoint[] = []
  songs.forEach((song) => {
    ;(['zc', 'zs', 'za'] as const).forEach((factor) => {
      const z = factor === 'zc' ? song.zcVector : factor === 'zs' ? song.zsVector : [song.zaVector[0], song.zaVector[1], 0]
      const base = new THREE.Vector3(z[0] * 0.35 + noise(), z[1] * 0.35 + noise(), z[2] * 0.35 + noise())
      const target = base.clone().add(direction[factor].clone().multiplyScalar(0.95))
      points.push({ song, factor, base, target, color: factorColor[factor] })
    })
  })
  return points
}

function ParticleCloud({ points }: { points: FactorPoint[] }) {
  const instancedRef = useRef<THREE.InstancedMesh>(null)
  const colorCache = useMemo(() => points.map((item) => item.color), [points])

  const separation = useSceneStore((state) => state.separation)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  const temp = useMemo(() => new THREE.Object3D(), [])

  useFrame((_state, delta) => {
    if (!instancedRef.current) return
    const instance = instancedRef.current
    const alpha = clamp(0.065 + delta * 0.1)

    points.forEach((item, index) => {
      const end = item.target.clone().multiplyScalar(separation)
      const start = item.base.clone().multiplyScalar(1 - separation)
      const pos = start.add(end)

      temp.position.lerp(pos, alpha)
      temp.scale.setScalar(item.factor === 'za' ? 0.09 : 0.08)
      temp.updateMatrix()
      instance.setMatrixAt(index, temp.matrix)
      instance.setColorAt(index, colorCache[index])
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
    setHoveredSongId(point.song.id)
  }

  const handleSelect = (event: ThreeEvent<MouseEvent>) => {
    if (event.instanceId == null) return
    const point = points[event.instanceId]
    if (!point) return
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
      <sphereGeometry args={[0.12, 12, 12]} />
      <meshStandardMaterial transparent opacity={0.9} emissive={new THREE.Color('#101828')} metalness={0.1} roughness={0.35} />
    </instancedMesh>
  )
}

function SceneCore() {
  const points = useMemo(() => buildFactorPoints(songPoints), [])

  return (
    <>
      <color attach="background" args={['#080912']} />
      <fog attach="fog" args={['#090b18', 4.5, 12]} />
      <ambientLight intensity={1.1} />
      <pointLight position={[4, 3, 2]} intensity={2.4} color={'#a55eea'} />
      <pointLight position={[-5, -2.5, 3]} intensity={1.8} color={'#4ecdc4'} />
      <ParticleCloud points={points} />
      <Stars radius={110} depth={32} count={1900} factor={4.2} saturation={0.65} fade speed={0.45} />
      <OrbitControls enablePan={false} enableZoom minDistance={2.4} maxDistance={7.2} autoRotate autoRotateSpeed={0.3} />
    </>
  )
}

export function LatentSpaceCanvas() {
  return (
    <Canvas camera={{ position: [0, 0.8, 5.2], fov: 52 }} dpr={[1, 1.6]} gl={{ antialias: true, alpha: true }}>
      <SceneCore />
    </Canvas>
  )
}
