'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Stars, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useSceneStore } from '@/components/state/scene-store'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { songPoints, type SongPoint } from '@/data/mock-data'
import { buildFactorMetrics } from '@/lib/factor-mapping'
import { motion, AnimatePresence } from 'framer-motion'

// Organic color palette - natural, flowing
const factorColors = {
  zc: new THREE.Color('#e85d4e'), // Warm coral - content/melody
  zs: new THREE.Color('#2d8a5e'), // Forest green - culture
  za: new THREE.Color('#4a90d9'), // Ocean blue - affect
  neutral: new THREE.Color('#f5f2eb') // Warm white
}

const factorGradients = {
  zc: ['#ff9a8b', '#e85d4e', '#c73e2e'],
  zs: ['#7dd3a8', '#2d8a5e', '#1a5233'],
  za: ['#8bc4ff', '#4a90d9', '#2a5a9e']
}

// Particle system for organic flow
function FlowParticles({ factor, active, intensity }: { factor: 'zc' | 'zs' | 'za'; active: boolean; intensity: number }) {
  const meshRef = useRef<THREE.Points>(null)
  const count = 200

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const vel = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 2 + Math.random() * 4
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)
      vel[i * 3] = (Math.random() - 0.5) * 0.01
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.01
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.01
    }
    return [pos, vel]
  }, [])

  useFrame((state) => {
    if (!meshRef.current) return
    const positions = meshRef.current.geometry.attributes.position.array as Float32Array
    const time = state.clock.elapsedTime

    for (let i = 0; i < count; i++) {
      const idx = i * 3
      const x = positions[idx]
      const y = positions[idx + 1]
      const z = positions[idx + 2]

      // Organic flow using noise-like movement
      const noiseX = Math.sin(time * 0.5 + y * 0.5) * 0.002
      const noiseY = Math.cos(time * 0.3 + x * 0.5) * 0.002
      const noiseZ = Math.sin(time * 0.4 + z * 0.5) * 0.002

      positions[idx] += velocities[idx] + noiseX
      positions[idx + 1] += velocities[idx + 1] + noiseY
      positions[idx + 2] += velocities[idx + 2] + noiseZ

      // Keep within bounds with soft return
      const dist = Math.sqrt(x * x + y * y + z * z)
      if (dist > 8) {
        positions[idx] *= 0.95
        positions[idx + 1] *= 0.95
        positions[idx + 2] *= 0.95
      }
    }
    meshRef.current.geometry.attributes.position.needsUpdate = true
  })

  const color = factorColors[factor]

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        color={color}
        transparent
        opacity={active ? 0.8 * intensity : 0.2}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

// Organic sphere representing a song
function SongOrb({
  song,
  position,
  isHovered,
  isSelected,
  onHover,
  onClick
}: {
  song: SongPoint
  position: THREE.Vector3
  isHovered: boolean
  isSelected: boolean
  onHover: (id: string | null) => void
  onClick: (id: string) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const metrics = useMemo(() => buildFactorMetrics(song), [song])
  // Color based on dominant factor
  const dominantFactor = useMemo(() => {
    const scores = { zc: metrics.zcStrength, zs: metrics.zsStrength, za: (metrics.zaArousal + 1) / 2 }
    return (Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0]) as 'zc' | 'zs' | 'za'
  }, [metrics])

  const baseColor = factorColors[dominantFactor]
  const emissiveIntensity = isSelected ? 0.8 : isHovered ? 0.5 : 0.2

  useFrame((state) => {
    if (!meshRef.current) return
    const time = state.clock.elapsedTime

    // Organic breathing animation
    const breathe = 1 + Math.sin(time * 2 + position.x * 0.5) * 0.1
    meshRef.current.scale.setScalar(isHovered ? 1.4 : isSelected ? 1.3 : 1)
    meshRef.current.scale.multiplyScalar(breathe)

    // Gentle rotation
    meshRef.current.rotation.y += 0.005
    meshRef.current.rotation.z += 0.003
  })

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerEnter={() => onHover(song.id)}
        onPointerLeave={() => onHover(null)}
        onClick={() => onClick(song.id)}
      >
        <sphereGeometry args={[0.25, 32, 32]} />
        <meshStandardMaterial
          color={baseColor}
          emissive={baseColor}
          emissiveIntensity={emissiveIntensity}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>

      {/* Glow halo */}
      <mesh scale={[2, 2, 2]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial
          color={baseColor}
          transparent
          opacity={isHovered ? 0.2 : isSelected ? 0.15 : 0.05}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Connection lines to factor axes when selected */}
      {(isHovered || isSelected) && (
        <>
          <Line
            points={[new THREE.Vector3(0, 0, 0), new THREE.Vector3(position.x * -0.5, 0, 0)]}
            color={factorColors.zc}
            lineWidth={1}
            transparent
            opacity={0.4}
          />
          <Line
            points={[new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, position.y * -0.5, 0)]}
            color={factorColors.zs}
            lineWidth={1}
            transparent
            opacity={0.4}
          />
          <Line
            points={[new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, position.z * -0.5)]}
            color={factorColors.za}
            lineWidth={1}
            transparent
            opacity={0.4}
          />
        </>
      )}

      {/* Label when hovered */}
      {isHovered && (
        <Html distanceFactor={10}>
          <div className="pointer-events-none whitespace-nowrap rounded-full border border-ink/15 bg-white/95 px-3 py-1.5 text-xs text-textMain shadow-lg backdrop-blur-sm">
            {song.title}
            <span className="ml-2 text-textSub">{song.culture}</span>
          </div>
        </Html>
      )}
    </group>
  )
}

// Organic factor axis - like a flowing vine or root
function OrganicAxis({
  factor,
  direction,
  active,
  separation,
  intensity
}: {
  factor: 'zc' | 'zs' | 'za'
  direction: THREE.Vector3
  active: boolean
  separation: number
  intensity: number
}) {
  const curve = useMemo(() => {
    const points: THREE.Vector3[] = []
    const length = 6 * (1 + separation * 0.5)
    const segments = 50

    for (let i = 0; i <= segments; i++) {
      const t = i / segments
      const basePos = direction.clone().multiplyScalar(t * length)

      // Add organic wiggle
      const wiggleX = Math.sin(t * Math.PI * 3) * 0.3 * (1 - t)
      const wiggleY = Math.cos(t * Math.PI * 2.5) * 0.2 * (1 - t)
      const wiggleZ = Math.sin(t * Math.PI * 4) * 0.25 * (1 - t)

      basePos.x += wiggleX
      basePos.y += wiggleY
      basePos.z += wiggleZ

      points.push(basePos)
    }

    return new THREE.CatmullRomCurve3(points)
  }, [direction, separation])

  const tubeRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (!tubeRef.current) return
    const time = state.clock.elapsedTime

    // Pulsing glow based on intensity
    const pulse = 1 + Math.sin(time * 3) * 0.1 * intensity
    tubeRef.current.scale.setScalar(pulse)
  })

  const color = factorColors[factor]

  return (
    <group>
      {/* Main axis tube */}
      <mesh ref={tubeRef}>
        <tubeGeometry args={[curve, 64, 0.08 * (active ? 1.2 : 0.8), 8, false]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={active ? 0.4 : 0.1}
          roughness={0.4}
          metalness={0.2}
          transparent
          opacity={active ? 0.9 : 0.5}
        />
      </mesh>

      {/* Glowing core */}
      {active && (
        <mesh>
          <tubeGeometry args={[curve, 64, 0.04, 8, false]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.6}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}

      {/* End capsule */}
      <mesh position={curve.getPoint(1)}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Label */}
      <Html position={curve.getPoint(1).add(new THREE.Vector3(0.5, 0.3, 0))} distanceFactor={8}>
        <div
          className="rounded-full px-3 py-1 text-xs font-semibold text-white"
          style={{
            background: `linear-gradient(135deg, ${factorGradients[factor][0]}, ${factorGradients[factor][1]})`,
            boxShadow: `0 4px 20px ${factorGradients[factor][1]}66`
          }}
        >
          {factor === 'zc' ? 'zc 内容' : factor === 'zs' ? 'zs 文化' : 'za 情感'}
        </div>
      </Html>
    </group>
  )
}

// Audio ripple effect
function AudioRipple({ position, color, active }: { position: THREE.Vector3; color: THREE.Color; active: boolean }) {
  const rings = useRef<THREE.Mesh[]>([])

  useFrame((state) => {
    if (!active) return
    const time = state.clock.elapsedTime

    rings.current.forEach((ring, i) => {
      if (!ring) return
      const offset = i * 0.5
      const scale = ((time * 2 + offset) % 3) + 0.5
      ring.scale.setScalar(scale)
      ;(ring.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 1 - (scale - 0.5) / 2.5) * 0.5
    })
  })

  if (!active) return null

  return (
    <group position={position}>
      {[0, 1, 2].map((i) => (
        <mesh key={i} ref={(el) => { if (el) rings.current[i] = el }}>
          <ringGeometry args={[0.3, 0.35, 32]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.5}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      ))}
    </group>
  )
}

// Main nebula scene
function NebulaScene({
  factorState,
  separation,
  cultureFilter
}: {
  factorState: { zc: boolean; zs: boolean; za: boolean }
  separation: number
  cultureFilter: string
}) {
  // Use separation to avoid unused variable warning
  void separation
  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const auditionFactor = useSceneStore((state) => state.auditionFactor)
  const setHoveredSongId = useSceneStore((state) => state.setHoveredSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)

  const filteredSongs = useMemo(() => {
    if (cultureFilter === 'All') return songPoints.slice(0, 24)
    return songPoints.filter(s => s.culture === cultureFilter).slice(0, 24)
  }, [cultureFilter])

  // Calculate song positions based on their vectors
  const songPositions = useMemo(() => {
    return filteredSongs.map(song => {
      const x = song.zcVector[0] * (2 + separation * 2)
      const y = song.zsVector[1] * (2 + separation * 2)
      const z = song.zaVector[0] * (2 + separation)
      return new THREE.Vector3(x, y, z)
    })
  }, [filteredSongs, separation])
  return (
    <>
      {/* Ambient particles */}
      <Stars radius={20} depth={50} count={500} factor={2} saturation={0.5} fade speed={0.5} />

      {/* Factor axes */}
      <OrganicAxis
        factor="zc"
        direction={new THREE.Vector3(1, 0, 0)}
        active={factorState.zc}
        separation={separation}
        intensity={auditionFactor === 'zc' ? 1 : 0.5}
      />
      <OrganicAxis
        factor="zs"
        direction={new THREE.Vector3(0, 1, 0)}
        active={factorState.zs}
        separation={separation}
        intensity={auditionFactor === 'zs' ? 1 : 0.5}
      />
      <OrganicAxis
        factor="za"
        direction={new THREE.Vector3(0, 0, 1)}
        active={factorState.za}
        separation={separation}
        intensity={auditionFactor === 'za' ? 1 : 0.5}
      />

      {/* Flow particles for each factor */}
      {factorState.zc && (
        <FlowParticles factor="zc" active={true} intensity={auditionFactor === 'zc' ? 1 : 0.3} />
      )}
      {factorState.zs && (
        <FlowParticles factor="zs" active={true} intensity={auditionFactor === 'zs' ? 1 : 0.3} />
      )}
      {factorState.za && (
        <FlowParticles factor="za" active={true} intensity={auditionFactor === 'za' ? 1 : 0.3} />
      )}

      {/* Song orbs */}
      {filteredSongs.map((song, index) => (
        <SongOrb
          key={song.id}
          song={song}
          position={songPositions[index]}
          isHovered={hoveredSongId === song.id}
          isSelected={selectedSongId === song.id}
          onHover={setHoveredSongId}
          onClick={setSelectedSongId}

        />
      ))}

      {/* Audio ripple at center when factor active */}
      {auditionFactor && (
        <AudioRipple
          position={new THREE.Vector3(0, 0, 0)}
          color={factorColors[auditionFactor]}
          active={!!auditionFactor}
        />
      )}

      {/* Soft ambient lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.5} color="#fff5eb" />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#e8f4f8" />

      {/* Controls */}
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={3}
        maxDistance={15}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </>
  )
}

// UI Overlay for the nebula
function NebulaUI({
  factorState,
  onToggleFactor,
  separation,
  onSeparationChange,
  cultureFilter,
  onCultureChange
}: {
  factorState: { zc: boolean; zs: boolean; za: boolean }
  onToggleFactor: (f: 'zc' | 'zs' | 'za') => void
  separation: number
  onSeparationChange: (v: number) => void
  cultureFilter: string
  onCultureChange: (c: string) => void
}) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const cultures = useMemo(() => ['All', ...Array.from(new Set(songPoints.map(s => s.culture)))], [])

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      {/* Top controls */}
      <div className="pointer-events-auto absolute left-4 right-4 top-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          {(['zc', 'zs', 'za'] as const).map(factor => (
            <button
              key={factor}
              onClick={() => onToggleFactor(factor)}
              className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-all duration-300 ${
                factorState[factor]
                  ? factor === 'zc'
                    ? 'border-zc/40 bg-zc/10 text-zc shadow-sm'
                    : factor === 'zs'
                    ? 'border-zs/40 bg-zs/10 text-zs shadow-sm'
                    : 'border-za/40 bg-za/10 text-za shadow-sm'
                  : 'border-ink/20 bg-white/80 text-textSub backdrop-blur-sm hover:bg-white'
              }`}
            >
              {factor === 'zc' ? (isZh ? 'zc 内容' : 'zc Content') : factor === 'zs' ? (isZh ? 'zs 文化' : 'zs Culture') : isZh ? 'za 情感' : 'za Affect'}
            </button>
          ))}
        </div>

        <select
          value={cultureFilter}
          onChange={(e) => onCultureChange(e.target.value)}
          className="rounded-full border border-ink/20 bg-white/80 px-3 py-1.5 text-xs text-textMain backdrop-blur-sm outline-none transition hover:bg-white"
        >
          {cultures.map(c => (
            <option key={c} value={c}>
              {c === 'All' ? (isZh ? '全部文化' : 'All Cultures') : c}
            </option>
          ))}
        </select>
      </div>

      {/* Bottom separation control */}
      <div className="pointer-events-auto absolute bottom-4 left-4 right-4">
        <div className="mx-auto max-w-md rounded-2xl border border-ink/15 bg-white/90 p-3 shadow-lg backdrop-blur-md">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-textMain">
              {isZh ? '因子分离度' : 'Factor Separation'}
            </span>
            <span className="font-mono text-xs text-textSub">{Math.round(separation * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={separation}
            onChange={(e) => onSeparationChange(parseFloat(e.target.value))}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink/20 accent-za outline-none"
          />
          <p className="mt-2 text-[10px] text-textSub">
            {isZh
              ? '拖动滑块观察因子轴的物理分离——模拟解纠缠过程'
              : 'Drag to observe physical separation of factor axes — simulating disentanglement'}
          </p>
        </div>
      </div>

      {/* Instructions */}
      <div className="pointer-events-none absolute left-4 top-1/2 hidden -translate-y-1/2 text-textSub md:block">
        <div className="space-y-1 text-[10px]">
          <p>🖱️ {isZh ? '拖拽旋转视角' : 'Drag to rotate'}</p>
          <p>🔍 {isZh ? '滚轮缩放' : 'Scroll to zoom'}</p>
          <p>👆 {isZh ? '点击球体选择' : 'Click orb to select'}</p>
        </div>
      </div>
    </div>
  )
}

// Main component
export function OrganicLatentNebula({
  className
}: {
  className?: string
}) {
  const [factorState, setFactorState] = useState({ zc: true, zs: true, za: true })
  const [separation, setSeparation] = useState(0)
  const [cultureFilter, setCultureFilter] = useState('All')
  const [isLoaded, setIsLoaded] = useState(false)

  const setSeparationStore = useSceneStore((state) => state.setSeparation)

  // Sync with store
  useEffect(() => {
    setSeparationStore(separation)
  }, [separation, setSeparationStore])

  useEffect(() => {
    const timer = setTimeout(() => setIsLoaded(true), 500)
    return () => clearTimeout(timer)
  }, [])

  const toggleFactor = (factor: 'zc' | 'zs' | 'za') => {
    setFactorState(prev => ({ ...prev, [factor]: !prev[factor] }))
  }

  return (
    <div className={`relative overflow-hidden rounded-3xl border border-ink/15 bg-white ${className}`}>
      {/* Loading state */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-20 flex items-center justify-center bg-white"
          >
            <div className="text-center">
              <div className="mb-4 h-8 w-8 animate-spin rounded-full border-2 border-ink/20 border-t-za" />
              <p className="text-xs text-textSub">Initializing Nebula...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [5, 5, 5], fov: 50 }}
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance'
        }}
      >
        <NebulaScene
          factorState={factorState}
          separation={separation}
          cultureFilter={cultureFilter}
        />
      </Canvas>

      {/* UI Overlay */}
      <NebulaUI
        factorState={factorState}
        onToggleFactor={toggleFactor}
        separation={separation}
        onSeparationChange={setSeparation}
        cultureFilter={cultureFilter}
        onCultureChange={setCultureFilter}
      />

      {/* Subtle border glow */}
      <div
        className="pointer-events-none absolute inset-0 rounded-3xl"
        style={{
          boxShadow: 'inset 0 0 60px rgba(0,0,0,0.05)'
        }}
      />
    </div>
  )
}




