'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

import { useAccessibility } from '@/components/providers/accessibility-provider'
import { useSceneStore } from '@/components/state/scene-store'
import { cn } from '@/lib/utils'

type Pulse = {
  id: number
  x: number
  y: number
  color: string
}

type FactorKey = 'zc' | 'zs' | 'za'

type Pad = {
  keyLabel: string
  note: string
  color: string
  factor: FactorKey
}

const pads: Pad[] = [
  { keyLabel: 'A', note: 'C4', color: '#ea4335', factor: 'zc' },
  { keyLabel: 'S', note: 'D4', color: '#ea4335', factor: 'zc' },
  { keyLabel: 'D', note: 'E4', color: '#188038', factor: 'zs' },
  { keyLabel: 'F', note: 'G4', color: '#188038', factor: 'zs' },
  { keyLabel: 'J', note: 'A4', color: '#1a73e8', factor: 'za' },
  { keyLabel: 'K', note: 'B4', color: '#1a73e8', factor: 'za' },
  { keyLabel: 'L', note: 'D5', color: '#1a73e8', factor: 'za' },
  { keyLabel: ';', note: 'E5', color: '#1a73e8', factor: 'za' }
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
  const clearFactorTimerRef = useRef<number | null>(null)

  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [audioReady, setAudioReady] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [ripples, setRipples] = useState<Pulse[]>([])

  const idRef = useRef(0)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const setAuditionFactor = useSceneStore((state) => state.setAuditionFactor)
  const auditionFactor = useSceneStore((state) => state.auditionFactor)

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

        masterGain.gain.value = 0.22
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

        oscillator.type = pad.factor === 'zc' ? 'triangle' : pad.factor === 'zs' ? 'square' : 'sine'
        oscillator.frequency.setValueAtTime(noteToFrequency(pad.note), now)

        gain.gain.setValueAtTime(0.0001, now)
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.34)

        oscillator.connect(gain)
        gain.connect(masterGainRef.current)

        oscillator.start(now)
        oscillator.stop(now + 0.35)

        setAuditionFactor(pad.factor)
        if (clearFactorTimerRef.current) {
          window.clearTimeout(clearFactorTimerRef.current)
        }
        clearFactorTimerRef.current = window.setTimeout(() => setAuditionFactor(null), 340)

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
    [ensureAudio, setAuditionFactor, spawnRipple]
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
      if (clearFactorTimerRef.current) {
        window.clearTimeout(clearFactorTimerRef.current)
      }
      setAuditionFactor(null)
      if (audioContextRef.current) {
        void audioContextRef.current.close()
        audioContextRef.current = null
        masterGainRef.current = null
      }
    }
  }, [keyMap, setAuditionFactor, triggerPad])

  const groups: Array<{ factor: FactorKey; label: string; color: string; keys: Pad[] }> = useMemo(
    () => [
      {
        factor: 'zc',
        label: isZh ? 'zc 内容' : 'zc content',
        color: '#ea4335',
        keys: pads.filter((item) => item.factor === 'zc')
      },
      {
        factor: 'zs',
        label: isZh ? 'zs 文化' : 'zs culture',
        color: '#188038',
        keys: pads.filter((item) => item.factor === 'zs')
      },
      {
        factor: 'za',
        label: isZh ? 'za 情感' : 'za affect',
        color: '#1a73e8',
        keys: pads.filter((item) => item.factor === 'za')
      }
    ],
    [isZh]
  )

  const factorMeta: Record<FactorKey, { label: string; color: string }> = {
    zc: { label: isZh ? 'zc 内容（Content）' : 'zc content', color: '#ea4335' },
    zs: { label: isZh ? 'zs 文化（Culture）' : 'zs culture', color: '#188038' },
    za: { label: isZh ? 'za 情感（Affect）' : 'za affect', color: '#1a73e8' }
  }

  const activeFactorText = auditionFactor ? factorMeta[auditionFactor].label : isZh ? '未激活（点击任意按键）' : 'inactive (click any pad)'
  const activeFactorColor = auditionFactor ? factorMeta[auditionFactor].color : '#6b7280'

  return (
    <div ref={wrapRef} className="relative overflow-hidden rounded-3xl paper-card p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="chapter-chip">{isZh ? '因子声音映射台（Factor Audio Mapper）' : 'factor audio mapper'}</span>
        <span className="sticker">{audioReady ? (isZh ? 'Audio Live' : 'audio live') : isZh ? 'Click Any Pad' : 'click any pad'}</span>
      </div>

      <p className="mb-2 text-sm text-textSub">
        {isZh
          ? '作用：通过键盘/点击触发 zc、zs、za 的试听通道，帮助你听见因子差异，并联动上方潜空间舞台。'
          : 'Purpose: trigger zc, zs, and za audition channels by keyboard/click to hear factor differences and sync with the stage above.'}
      </p>

      <div className="mb-2 rounded-xl border border-ink/15 bg-white/85 px-3 py-2 text-xs text-textSub">
        <span className="font-semibold text-textMain">{isZh ? '当前激活：' : 'Active channel:'}</span>
        <span className="ml-2 inline-flex items-center font-semibold" style={{ color: activeFactorColor }}>
          <span className="mr-1 inline-flex h-2 w-2 rounded-full" style={{ background: activeFactorColor }} />
          {activeFactorText}
        </span>
      </div>

      <div className="mb-2 grid gap-2 md:grid-cols-3">
        {groups.map((group) => (
          <div key={group.factor} className={cn('rounded-xl border px-2 py-1.5 text-[11px] font-semibold transition', auditionFactor === group.factor ? 'border-ink/30 bg-white text-textMain' : 'border-ink/15 bg-white/70 text-textSub')}>
            <span className="inline-flex h-2 w-2 rounded-full" style={{ background: group.color }} />
            <span className="ml-1">{group.label}</span>
          </div>
        ))}
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
                <div className="mt-1 text-[10px] text-textSub">{pad.factor === 'zc' ? (isZh ? 'zc 内容' : 'zc content') : pad.factor === 'zs' ? (isZh ? 'zs 文化' : 'zs culture') : isZh ? 'za 情感' : 'za affect'}</div>
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



