'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useMemo, useRef } from 'react'
import { ArrowDownRight, Sparkle } from 'lucide-react'

import { songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { useMediaQuery } from '@/hooks/use-media-query'

const LatentSpaceCanvas = dynamic(
  () => import('@/components/visuals/latent-space-canvas').then((mod) => mod.LatentSpaceCanvas),
  {
    ssr: false,
    loading: () => <div className="h-[62vh] animate-pulse rounded-3xl bg-white/10" />
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
    setSeparation(Math.min(1, value * 1.35))
  })

  return (
    <section
      id="hero"
      ref={sectionRef}
      data-section-id="hero"
      className="relative min-h-screen overflow-hidden px-4 pt-28 md:px-10"
      aria-labelledby="hero-title"
    >
      <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-[1.05fr_1fr]">
        <motion.div
          className="reveal-item space-y-6"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-zs/40 bg-zs/10 px-3 py-1 text-xs font-mono tracking-wide text-zs">
            <Sparkle size={13} />
            Deep Disentanglement Representation Learning
          </div>
          <h1 id="hero-title" className="font-display text-4xl font-black leading-tight text-textMain md:text-6xl lg:text-7xl">
            {title}
          </h1>
          <p className="max-w-2xl font-body text-lg leading-relaxed text-textSub md:text-xl">{lead}</p>
          <p className="font-mono text-sm text-textSub/90">{hint}</p>

          <div className="flex flex-wrap gap-3 pt-3">
            <button
              onClick={() => onNavigate('galaxy')}
              className="group rounded-full bg-zs px-5 py-2.5 font-semibold text-abyss transition hover:-translate-y-0.5 hover:shadow-neon"
            >
              {ctaPrimary}
              <ArrowDownRight size={16} className="ml-2 inline transition group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
            </button>
            <button
              onClick={() => onNavigate('lab')}
              className="rounded-full border border-white/20 bg-white/5 px-5 py-2.5 font-semibold text-textMain transition hover:border-zc/60 hover:bg-zc/10"
            >
              {ctaSecondary}
            </button>
          </div>

          {hoveredSong ? (
            <div className="max-w-xl animate-floatSlow">
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

        <div className="relative reveal-item">
          <div className="absolute -inset-6 -z-10 rounded-[3rem] bg-gradient-to-r from-zc/20 via-za/15 to-zs/20 blur-3xl" />
          <div className="h-[62vh] overflow-hidden rounded-3xl border border-white/10 bg-black/30 shadow-neon md:h-[70vh]">
            {isMobile ? (
              <div className="flex h-full items-center justify-center px-6 text-center">
                <p className="max-w-md font-body text-base text-textSub">
                  Mobile preview mode: 3D latent space is simplified for performance. Rotate and deep interaction are enabled on tablet/desktop.
                </p>
              </div>
            ) : (
              <LatentSpaceCanvas />
            )}
          </div>
          <div className="mt-3 rounded-xl border border-white/10 bg-black/30 p-3 font-mono text-xs text-textSub">
            Latent factor split progress is driven by scroll. Current separation: dynamic from 0 to 1.
          </div>
        </div>
      </div>

      <p className="sr-only">
        Three-dimensional latent space map: particles represent songs and disentangled factors for content, culture style, and emotion.
      </p>
    </section>
  )
}
