'use client'

import dynamic from 'next/dynamic'
import { motion, useMotionValueEvent, useScroll } from 'framer-motion'
import { useMemo, useRef } from 'react'
import { ArrowDownRight, Keyboard, Orbit, Pointer, ScrollText } from 'lucide-react'

import { songPoints } from '@/data/mock-data'
import { useSceneStore } from '@/components/state/scene-store'
import { SongRadar } from '@/components/visuals/song-radar'
import { PulseConsole } from '@/components/visuals/pulse-console'
import { useMediaQuery } from '@/hooks/use-media-query'

const LatentSpaceCanvas = dynamic(() => import('@/components/visuals/latent-space-canvas').then((mod) => mod.LatentSpaceCanvas), {
  ssr: false,
  loading: () => <div className="h-[56vh] animate-pulse rounded-3xl bg-white/80" />
})

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

  const hoveredSong = useMemo(() => songPoints.find((song) => song.id === (selectedSongId ?? hoveredSongId)) ?? null, [hoveredSongId, selectedSongId])

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start']
  })

  useMotionValueEvent(scrollYProgress, 'change', (value) => {
    setSeparation(Math.min(1, value * 1.35))
  })

  return (
    <section id="hero" ref={sectionRef} data-section-id="hero" className="relative min-h-screen overflow-hidden px-4 pb-10 pt-28 md:px-10" aria-labelledby="hero-title">
      <div className="mx-auto max-w-7xl">
        <div className="mb-5 flex flex-wrap items-center gap-2 reveal-item">
          <span className="sticker">interactive research website</span>
          <span className="sticker">chapter 00 / prelude</span>
          <span className="sticker">ddrl + ot + pal</span>
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
          <motion.div className="reveal-item space-y-5" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65 }}>
            <h1 id="hero-title" className="hero-title-glow font-display text-5xl font-semibold leading-[0.92] text-textMain md:text-7xl xl:text-8xl">
              {title}
            </h1>
            <p className="max-w-xl text-base leading-relaxed text-textSub md:text-lg">{lead}</p>

            <div className="grid gap-2 md:grid-cols-3">
              {[
                { icon: ScrollText, title: 'Narrate', text: 'Scroll chapters as an editorial story.' },
                { icon: Orbit, title: 'Orbit', text: 'Rotate latent constellations in 3D.' },
                { icon: Keyboard, title: 'Play', text: 'Trigger sonic pulses with keyboard pads.' }
              ].map((item) => (
                <div key={item.title} className="note-card">
                  <item.icon size={14} className="text-zs" />
                  <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.13em] text-textSub">{item.title}</p>
                  <p className="mt-1 text-sm text-textSub">{item.text}</p>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-3 pt-1">
              <button onClick={() => onNavigate('galaxy')} className="group rounded-full bg-zs px-5 py-2.5 font-semibold text-white transition hover:-translate-y-0.5">
                {ctaPrimary}
                <ArrowDownRight size={16} className="ml-2 inline transition group-hover:translate-x-0.5 group-hover:translate-y-0.5" />
              </button>
              <button onClick={() => onNavigate('lab')} className="rounded-full border border-ink/25 bg-white px-5 py-2.5 font-semibold text-textMain transition hover:border-zc/60 hover:bg-zc/10">
                {ctaSecondary}
              </button>
            </div>

            <div className="rounded-2xl border border-ink/15 bg-white/80 p-3">
              <div className="flex items-center gap-2 text-xs text-textSub">
                <Pointer size={12} />
                <span className="font-mono uppercase tracking-[0.12em]">interaction note</span>
              </div>
              <p className="mt-1 text-sm text-textSub">{hint}</p>
            </div>

            {hoveredSong ? (
              <div className="max-w-xl">
                <SongRadar song={hoveredSong} />
                {selectedSongId ? (
                  <button className="mt-2 text-xs text-textSub underline decoration-dotted underline-offset-4" onClick={() => setSelectedSongId(null)}>
                    Clear selected anatomy view
                  </button>
                ) : null}
              </div>
            ) : null}
          </motion.div>

          <div className="reveal-item space-y-4">
            <div className="relative overflow-hidden rounded-3xl paper-card scanline">
              <div className="absolute left-3 top-3 z-10 chapter-chip">latent space atlas</div>
              <div className="h-[56vh] md:h-[62vh]">
                {isMobile ? (
                  <div className="flex h-full items-center justify-center px-6 text-center">
                    <p className="max-w-md text-base text-textSub">Mobile mode shows a simplified preview. Desktop unlocks full 3D interaction and point inspection.</p>
                  </div>
                ) : (
                  <LatentSpaceCanvas />
                )}
              </div>
            </div>
            <PulseConsole />
          </div>
        </div>
      </div>

      <p className="sr-only">Three-dimensional latent map with disentangled content, style, and affect trajectories.</p>
    </section>
  )
}