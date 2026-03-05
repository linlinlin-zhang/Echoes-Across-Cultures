'use client'

import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Canvas, ThreeEvent, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

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
  zc: new THREE.Color('#ea4335'),
  zs: new THREE.Color('#188038'),
  za: new THREE.Color('#1a73e8')
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
    zc: new THREE.Vector3(1.65, -0.72, 0.38),
    zs: new THREE.Vector3(-1.48, 0.62, 1.18),
    za: new THREE.Vector3(0.2, 1.64, -1.42)
  }

  const points: FactorPoint[] = []
  songs.forEach((song) => {
    ;(['zc', 'zs', 'za'] as const).forEach((factor) => {
      const z = factor === 'zc' ? song.zcVector : factor === 'zs' ? song.zsVector : [song.zaVector[0], song.zaVector[1], 0]
      const base = new THREE.Vector3(z[0] * 0.34 + noise(), z[1] * 0.34 + noise(), z[2] * 0.34 + noise())
      const target = base.clone().add(direction[factor].clone().multiplyScalar(0.98))
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
    const alpha = clamp(0.07 + delta * 0.1)

    points.forEach((item, index) => {
      const end = item.target.clone().multiplyScalar(separation)
      const start = item.base.clone().multiplyScalar(1 - separation)
      const pos = start.add(end)

      temp.position.lerp(pos, alpha)
      temp.scale.setScalar(item.factor === 'za' ? 0.092 : 0.082)
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
      <sphereGeometry args={[0.11, 12, 12]} />
      <meshStandardMaterial transparent opacity={0.88} emissive={new THREE.Color('#f6e9cf')} metalness={0.05} roughness={0.28} />
    </instancedMesh>
  )
}

function SceneCore() {
  const points = useMemo(() => buildFactorPoints(songPoints), [])

  return (
    <>
      <color attach="background" args={['#f8efdc']} />
      <fog attach="fog" args={['#efe1c5', 4.5, 12]} />
      <ambientLight intensity={1.4} />
      <pointLight position={[4, 3, 2]} intensity={2} color={'#ea4335'} />
      <pointLight position={[-5, -2.5, 3]} intensity={1.8} color={'#188038'} />
      <pointLight position={[0, 3.2, -2]} intensity={1.5} color={'#1a73e8'} />
      <ParticleCloud points={points} />
      <OrbitControls enablePan={false} enableZoom minDistance={2.4} maxDistance={7.2} autoRotate autoRotateSpeed={0.2} />
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

