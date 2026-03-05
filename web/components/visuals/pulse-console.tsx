'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import * as Tone from 'tone'

type Pulse = {
  id: number
  x: number
  y: number
  color: string
}

type Pad = {
  keyLabel: string
  note: string
  color: string
  hint: string
}

const pads: Pad[] = [
  { keyLabel: 'A', note: 'C4', color: '#ff6f61', hint: 'content spark' },
  { keyLabel: 'S', note: 'D4', color: '#f29f05', hint: 'motif rise' },
  { keyLabel: 'D', note: 'E4', color: '#00a7a0', hint: 'culture drift' },
  { keyLabel: 'F', note: 'G4', color: '#3d77c3', hint: 'flow pulse' },
  { keyLabel: 'J', note: 'A4', color: '#7e57c2', hint: 'affect lift' },
  { keyLabel: 'K', note: 'B4', color: '#b65bd8', hint: 'memory shade' },
  { keyLabel: 'L', note: 'D5', color: '#ff5ea8', hint: 'bridge jump' },
  { keyLabel: ';', note: 'E5', color: '#54b95a', hint: 'serendipity' }
]

export function PulseConsole() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const synthRef = useRef<Tone.PolySynth | null>(null)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [audioReady, setAudioReady] = useState(false)
  const [ripples, setRipples] = useState<Pulse[]>([])
  const idRef = useRef(0)

  const keyMap = useMemo(() => new Map(pads.map((pad) => [pad.keyLabel, pad])), [])

  const ensureAudio = useCallback(async () => {
    if (audioReady) return
    await Tone.start()
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle8' },
      envelope: { attack: 0.02, decay: 0.2, sustain: 0.1, release: 0.35 }
    }).toDestination()
    synth.volume.value = -12
    synthRef.current = synth
    setAudioReady(true)
  }, [audioReady])

  const spawnRipple = useCallback((x: number, y: number, color: string) => {
    const id = ++idRef.current
    setRipples((prev) => [...prev, { id, x, y, color }].slice(-16))
    window.setTimeout(() => setRipples((prev) => prev.filter((item) => item.id !== id)), 600)
  }, [])

  const triggerPad = useCallback(
    async (pad: Pad, point?: { x: number; y: number }) => {
      await ensureAudio()
      synthRef.current?.triggerAttackRelease(pad.note, '8n')
      setActiveKey(pad.keyLabel)
      window.setTimeout(() => setActiveKey((prev) => (prev === pad.keyLabel ? null : prev)), 130)

      const rect = wrapRef.current?.getBoundingClientRect()
      if (rect) {
        const x = point?.x ?? rect.width * (0.12 + Math.random() * 0.76)
        const y = point?.y ?? rect.height * (0.22 + Math.random() * 0.58)
        spawnRipple(x, y, pad.color)
      }
    },
    [ensureAudio, spawnRipple]
  )

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toUpperCase()
      const matched = keyMap.get(key)
      if (!matched) return
      event.preventDefault()
      triggerPad(matched)
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      synthRef.current?.dispose()
    }
  }, [keyMap, triggerPad])

  return (
    <div ref={wrapRef} className="relative overflow-hidden rounded-3xl paper-card p-4 scanline">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <span className="chapter-chip">interactive latent instrument</span>
          <p className="mt-2 font-display text-lg text-textMain">Tap pads or press keyboard keys</p>
        </div>
        <span className="sticker">{audioReady ? 'audio live' : 'click any pad'}</span>
      </div>

      <div className="grid grid-cols-4 gap-2 md:gap-3">
        {pads.map((pad) => {
          const active = activeKey === pad.keyLabel
          return (
            <button
              key={pad.keyLabel}
              onClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                triggerPad(pad, {
                  x: rect.left + rect.width / 2 - (wrapRef.current?.getBoundingClientRect().left ?? 0),
                  y: rect.top + rect.height / 2 - (wrapRef.current?.getBoundingClientRect().top ?? 0)
                })
              }}
              className="group relative overflow-hidden rounded-2xl border border-ink/20 bg-white p-3 text-left transition duration-150 hover:border-ink/45"
              style={{ boxShadow: active ? `0 0 0 1px ${pad.color}, 0 0 20px ${pad.color}77` : undefined }}
            >
              <div className="absolute inset-0 opacity-35 transition duration-150 group-hover:opacity-60" style={{ background: `radial-gradient(circle at 50% 20%, ${pad.color}, transparent 70%)` }} />
              <div className="relative">
                <div className="font-display text-xl font-bold" style={{ color: pad.color }}>{pad.keyLabel}</div>
                <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.15em] text-textSub">{pad.note}</div>
                <div className="mt-2 text-xs text-textSub">{pad.hint}</div>
              </div>
            </button>
          )
        })}
      </div>

      <AnimatePresence>
        {ripples.map((ripple) => (
          <motion.span
            key={ripple.id}
            initial={{ scale: 0.1, opacity: 0.9 }}
            animate={{ scale: 3.8, opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="pointer-events-none absolute h-16 w-16 rounded-full"
            style={{ left: ripple.x - 32, top: ripple.y - 32, border: `2px solid ${ripple.color}`, boxShadow: `0 0 24px ${ripple.color}` }}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}