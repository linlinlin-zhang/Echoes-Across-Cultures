'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useMemo, useRef } from 'react'
import { ArrowDownRight, Compass, Joystick, Orbit } from 'lucide-react'

import { songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { PulseConsole } from '@/components/visuals/pulse-console'
import { useMediaQuery } from '@/hooks/use-media-query'

const LatentSpaceCanvas = dynamic(
  () => import('@/components/visuals/latent-space-canvas').then((mod) => mod.LatentSpaceCanvas),
  {
    ssr: false,
    loading: () => <div className="h-[58vh] animate-pulse rounded-3xl bg-white/10" />
  }
)

type HeroSectionProps = {
  title: string
  lead: string
  hint: string
  ctaPrimary: string
  ctaSecondary: string
  onNavigate: (id: 'galaxy' | 'lab') => void
}

export function HeroSection({ title, lead, hint, ctaPrimary, ctaSecondary, onNavigate }: HeroSectionProps) {
  const sectionRef = useRef<HTMLElement>(null)
  const isMobile = useMediaQuery('(max-width: 767px)')

  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)
  const setSelectedSongId = useSceneStore((state) => state.setSelectedSongId)
  const setSeparation = useSceneStore((state) => state.setSeparation)

  const hoveredSong = useMemo(
    () => songPoints.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? null,
    [hoveredSongId, selectedSongId]
  )

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setSeparation(Math.min(1, value * 1.45))
  })

  return (
    <section
      id="hero"
      ref={sectionRef}
      data-section-id="hero"
      className="relative min-h-screen overflow-hidden px-4 pb-8 pt-28 md:px-10"
      aria-labelledby="hero-title"
    >
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <motion.div
          className="reveal-item space-y-5"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-white/20 panel-glass px-3 py-1 font-mono text-[11px] uppercase tracking-[0.24em] text-zs">
            <Orbit size={12} />
            Cross-Cultural Cognitive Music Interface
          </div>

          <h1 id="hero-title" className="hero-title-glow font-display text-4xl font-extrabold leading-[0.95] text-textMain md:text-6xl lg:text-7xl">
            {title}
          </h1>
          <p className="max-w-2xl font-body text-base leading-relaxed text-textSub md:text-lg">{lead}</p>

          <div className="grid gap-2 rounded-2xl panel-deep p-4 md:grid-cols-3">
            {[
              { icon: Compass, label: 'Navigate', text: 'Scroll to split latent factors' },
              { icon: Joystick, label: 'Play', text: 'Keyboard pads trigger audiovisual pulses' },
              { icon: Orbit, label: 'Inspect', text: 'Hover points to open anatomy cards' }
            ].map((item) => (
              <div key={item.label} className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                <item.icon size={14} className="text-zs" />
                <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">{item.label}</p>
                <p className="mt-1 text-xs text-textSub">{item.text}</p>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-3 pt-1">
            <button
              onClick={() => onNavigate('galaxy')}
              className="group rounded-full bg-zs px-5 py-2.5 font-semibold text-abyss transition hover:-translate-y-0.5 hover:shadow-[0_0_22px_rgba(78,205,196,0.55)]"
            >
              {ctaPrimary}
              <ArrowDownRight size={16} className="ml-2 inline transition group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
            </button>
            <button
              onClick={() => onNavigate('lab')}
              className="rounded-full border border-white/25 bg-white/5 px-5 py-2.5 font-semibold text-textMain transition hover:border-zc/60 hover:bg-zc/10"
            >
              {ctaSecondary}
            </button>
          </div>

          <p className="font-mono text-xs text-textSub/90">{hint}</p>

          {hoveredSong ? (
            <div className="max-w-xl">
              <SongRadar song={hoveredSong} />
              {selectedSongId ? (
                <button
                  className="mt-2 text-xs text-textSub underline decoration-dotted underline-offset-4"
                  onClick={() => setSelectedSongId(null)}
                >
                  Clear selected anatomy view
                </button>
              ) : null}
            </div>
          ) : null}
        </motion.div>

        <div className="reveal-item space-y-4">
          <div className="relative overflow-hidden rounded-3xl panel-deep scanline">
            <div className="absolute left-3 top-3 z-10 rounded-full border border-white/20 bg-black/50 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-textSub">
              Latent Orbit View
            </div>
            <div className="h-[58vh] md:h-[64vh]">
              {isMobile ? (
                <div className="flex h-full items-center justify-center px-6 text-center">
                  <p className="max-w-md font-body text-base text-textSub">
                    Mobile mode uses a lightweight preview. Open desktop for full 3D navigation, point-level selection, and orbit control.
                  </p>
                </div>
              ) : (
                <LatentSpaceCanvas />
              )}
            </div>
          </div>

          <PulseConsole />
        </div>
      </div>

      <p className="sr-only">Three-dimensional latent map with disentangled content, style, and affect trajectories.</p>
    </section>
  )
}