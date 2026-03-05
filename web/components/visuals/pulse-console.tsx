'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

import { useAccessibility } from '@/components/providers/accessibility-provider'

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
}

const pads: Pad[] = [
  { keyLabel: 'A', note: 'C4', color: '#ea4335' },
  { keyLabel: 'S', note: 'D4', color: '#fbbc04' },
  { keyLabel: 'D', note: 'E4', color: '#188038' },
  { keyLabel: 'F', note: 'G4', color: '#4285f4' },
  { keyLabel: 'J', note: 'A4', color: '#1a73e8' },
  { keyLabel: 'K', note: 'B4', color: '#a142f4' },
  { keyLabel: 'L', note: 'D5', color: '#e52592' },
  { keyLabel: ';', note: 'E5', color: '#34a853' }
]

const semitoneMap: Record<string, number> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11
}

function noteToFrequency(note: string) {
  const match = note.match(/^([A-G])([#b]?)([0-8])$/)
  if (!match) return 261.63

  const key = `${match[1]}${match[2]}`
  const semitone = semitoneMap[key]
  if (semitone == null) return 261.63

  const octave = Number(match[3])
  const midi = (octave + 1) * 12 + semitone
  return 440 * Math.pow(2, (midi - 69) / 12)
}

export function PulseConsole() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const masterGainRef = useRef<GainNode | null>(null)

  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [audioReady, setAudioReady] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [ripples, setRipples] = useState<Pulse[]>([])

  const idRef = useRef(0)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const keyMap = useMemo(() => new Map(pads.map((pad) => [pad.keyLabel, pad])), [])

  const ensureAudio = useCallback(async () => {
    try {
      const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!AudioContextCtor) {
        setAudioError(isZh ? '浏览器不支持 AudioContext。' : 'AudioContext is not supported in this browser.')
        return false
      }

      if (!audioContextRef.current) {
        const context = new AudioContextCtor()
        const masterGain = context.createGain()
        const compressor = context.createDynamicsCompressor()

        masterGain.gain.value = 0.2
        compressor.threshold.value = -20
        compressor.knee.value = 20
        compressor.ratio.value = 8
        compressor.attack.value = 0.003
        compressor.release.value = 0.2

        masterGain.connect(compressor)
        compressor.connect(context.destination)

        audioContextRef.current = context
        masterGainRef.current = masterGain
      }

      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }

      setAudioReady(true)
      setAudioError(null)
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setAudioReady(false)
      setAudioError(message)
      return false
    }
  }, [isZh])

  const spawnRipple = useCallback((x: number, y: number, color: string) => {
    const id = ++idRef.current
    setRipples((prev) => [...prev, { id, x, y, color }].slice(-16))
    window.setTimeout(() => setRipples((prev) => prev.filter((item) => item.id !== id)), 600)
  }, [])

  const triggerPad = useCallback(
    async (pad: Pad, point?: { x: number; y: number }) => {
      try {
        const ready = await ensureAudio()
        if (!ready || !audioContextRef.current || !masterGainRef.current) return

        const context = audioContextRef.current
        const now = context.currentTime

        const oscillator = context.createOscillator()
        const gain = context.createGain()

        oscillator.type = 'triangle'
        oscillator.frequency.setValueAtTime(noteToFrequency(pad.note), now)

        gain.gain.setValueAtTime(0.0001, now)
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.34)

        oscillator.connect(gain)
        gain.connect(masterGainRef.current)

        oscillator.start(now)
        oscillator.stop(now + 0.35)

        setActiveKey(pad.keyLabel)
        window.setTimeout(() => setActiveKey((prev) => (prev === pad.keyLabel ? null : prev)), 130)

        const rect = wrapRef.current?.getBoundingClientRect()
        if (rect) {
          const x = point?.x ?? rect.width * (0.12 + Math.random() * 0.76)
          const y = point?.y ?? rect.height * (0.22 + Math.random() * 0.58)
          spawnRipple(x, y, pad.color)
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        setAudioError(message)
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
      void triggerPad(matched)
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      if (audioContextRef.current) {
        void audioContextRef.current.close()
        audioContextRef.current = null
        masterGainRef.current = null
      }
    }
  }, [keyMap, triggerPad])

  return (
    <div ref={wrapRef} className="relative overflow-hidden rounded-3xl paper-card p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="chapter-chip">{isZh ? '交互潜变量乐器' : 'interactive latent instrument'}</span>
        <span className="sticker">{audioReady ? (isZh ? 'Audio Live' : 'audio live') : isZh ? 'Click Any Pad' : 'click any pad'}</span>
      </div>

      <div className="grid grid-cols-4 gap-2 md:gap-3">
        {pads.map((pad) => {
          const active = activeKey === pad.keyLabel
          return (
            <button
              key={pad.keyLabel}
              onClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                void triggerPad(pad, {
                  x: rect.left + rect.width / 2 - (wrapRef.current?.getBoundingClientRect().left ?? 0),
                  y: rect.top + rect.height / 2 - (wrapRef.current?.getBoundingClientRect().top ?? 0)
                })
              }}
              aria-label={`Pad ${pad.keyLabel}`}
              className="group relative overflow-hidden rounded-2xl border border-ink/20 bg-white p-3 text-left transition duration-150 hover:border-ink/45"
              style={{ boxShadow: active ? `0 0 0 1px ${pad.color}, 0 0 20px ${pad.color}77` : undefined }}
            >
              <div className="absolute inset-0 opacity-35 transition duration-150 group-hover:opacity-60" style={{ background: `radial-gradient(circle at 50% 20%, ${pad.color}, transparent 70%)` }} />
              <div className="relative">
                <div className="font-display text-xl font-bold" style={{ color: pad.color }}>
                  {pad.keyLabel}
                </div>
                <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.15em] text-textSub">{pad.note}</div>
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

      {audioError ? <div className="mt-3 rounded-xl border border-zc/35 bg-zc/10 px-3 py-2 text-xs text-zc">{audioError}</div> : null}
    </div>
  )
}
