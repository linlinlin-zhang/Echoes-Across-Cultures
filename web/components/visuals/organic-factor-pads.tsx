'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { useSceneStore } from '@/components/state/scene-store'
import { songPoints } from '@/data/mock-data'
import { buildFactorMetrics, factorSummaryEn, factorSummaryZh } from '@/lib/factor-mapping'
import { Volume2, Music, Globe, Heart } from 'lucide-react'

type FactorKey = 'zc' | 'zs' | 'za'

interface Pad {
  id: string
  factor: FactorKey
  note: string
  octave: number
}

// Generate organic pad layout
function generatePads(): Pad[] {
  const pads: Pad[] = []
  const factors: FactorKey[] = ['zc', 'zs', 'za']
  const notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

  factors.forEach((factor) => {
    for (let i = 0; i < 4; i++) {
      pads.push({
        id: `${factor}-${i}`,
        factor,
        note: notes[i % notes.length],
        octave: 4 + Math.floor(i / 2)
      })
    }
  })

  return pads
}

// Audio synthesis
function playOrganicTone({
  context,
  factor,
  frequency,
  metrics
}: {
  context: AudioContext
  factor: FactorKey
  frequency: number
  metrics: ReturnType<typeof buildFactorMetrics>
}) {
  const masterGain = context.createGain()
  masterGain.gain.value = 0.3
  masterGain.connect(context.destination)

  if (factor === 'zc') {
    // Melodic/arpeggio for content
    const ratios = metrics.melodySteps.slice(0, 4).map((v) => 0.82 + v * 0.7)
    ratios.forEach((ratio, i) => {
      const osc = context.createOscillator()
      const gain = context.createGain()
      osc.type = 'triangle'
      osc.frequency.value = frequency * ratio
      osc.connect(gain)
      gain.connect(masterGain)

      const now = context.currentTime
      gain.gain.setValueAtTime(0, now + i * 0.1)
      gain.gain.linearRampToValueAtTime(0.1, now + i * 0.1 + 0.05)
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.3)

      osc.start(now + i * 0.1)
      osc.stop(now + i * 0.1 + 0.35)
    })
  } else if (factor === 'zs') {
    // Cultural timbre
    const osc = context.createOscillator()
    const gain = context.createGain()
    const filter = context.createBiquadFilter()

    osc.type = metrics.wave
    osc.frequency.value = frequency * 0.95

    filter.type = 'bandpass'
    filter.frequency.value = metrics.filterHz
    filter.Q.value = 0.85

    osc.connect(filter)
    filter.connect(gain)
    gain.connect(masterGain)

    const now = context.currentTime
    gain.gain.setValueAtTime(0, now)
    gain.gain.linearRampToValueAtTime(0.15, now + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5)

    osc.start(now)
    osc.stop(now + 0.55)
  } else {
    // Affective - emotional envelope
    const osc = context.createOscillator()
    const gain = context.createGain()
    const vibrato = context.createOscillator()
    const vibratoGain = context.createGain()

    osc.type = 'sine'
    osc.frequency.value = frequency

    vibrato.type = 'sine'
    vibrato.frequency.value = metrics.vibratoHz
    vibratoGain.gain.value = 8

    vibrato.connect(vibratoGain)
    vibratoGain.connect(osc.frequency)
    osc.connect(gain)
    gain.connect(masterGain)

    const now = context.currentTime
    gain.gain.setValueAtTime(0, now)
    gain.gain.exponentialRampToValueAtTime(0.12, now + metrics.attack)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3 + metrics.release)

    osc.start(now)
    vibrato.start(now)
    osc.stop(now + 0.4 + metrics.release)
    vibrato.stop(now + 0.4 + metrics.release)
  }
}

function noteToFrequency(note: string, octave: number): number {
  const semitoneMap: Record<string, number> = {
    C: 0, 'C#': 1, D: 2, 'D#': 3, E: 4, F: 5, 'F#': 6, G: 7, 'G#': 8, A: 9, 'A#': 10, B: 11
  }
  const semitone = semitoneMap[note] ?? 0
  const midi = (octave + 1) * 12 + semitone
  return 440 * Math.pow(2, (midi - 69) / 12)
}

export function OrganicFactorPads({ className }: { className?: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const [pads] = useState(() => generatePads())
  const [activePads, setActivePads] = useState<Set<string>>(new Set())
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
  const [audioEnabled, setAudioEnabled] = useState(false)

  const setAuditionFactor = useSceneStore((state) => state.setAuditionFactor)
  const setAuditionTrace = useSceneStore((state) => state.setAuditionTrace)
  const hoveredSongId = useSceneStore((state) => state.hoveredSongId)
  const selectedSongId = useSceneStore((state) => state.selectedSongId)

  const activeSong = useMemo(() => {
    return songPoints.find((s) => s.id === (selectedSongId ?? hoveredSongId)) ?? songPoints[0]
  }, [hoveredSongId, selectedSongId])

  const metrics = useMemo(() => buildFactorMetrics(activeSong), [activeSong])

  const initAudio = useCallback(async () => {
    if (audioContext) return
    const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextCtor) return

    const ctx = new AudioContextCtor()
    await ctx.resume()
    setAudioContext(ctx)
    setAudioEnabled(true)
  }, [audioContext])

  const triggerPad = useCallback(
    async (pad: Pad) => {
      await initAudio()

      setActivePads((prev) => {
        const next = new Set(prev)
        next.add(pad.id)
        setTimeout(() => {
          setActivePads((current) => {
            const updated = new Set(current)
            updated.delete(pad.id)
            return updated
          })
        }, 300)
        return next
      })

      if (audioContext) {
        const frequency = noteToFrequency(pad.note, pad.octave)
        playOrganicTone({ context: audioContext, factor: pad.factor, frequency, metrics })
      }

      setAuditionFactor(pad.factor)
      setAuditionTrace({
        factor: pad.factor,
        summaryZh: factorSummaryZh(pad.factor, activeSong, metrics),
        summaryEn: factorSummaryEn(pad.factor, activeSong, metrics),
        timestamp: Date.now()
      })

      setTimeout(() => setAuditionFactor(null), 500)
    },
    [audioContext, metrics, activeSong, setAuditionFactor, setAuditionTrace, initAudio]
  )

  // Keyboard shortcuts
  useEffect(() => {
    const keyMap: Record<string, string> = {
      a: 'zc-0',
      s: 'zc-1',
      d: 'zc-2',
      f: 'zc-3',
      j: 'zs-0',
      k: 'zs-1',
      l: 'zs-2',
      ';': 'zs-3',
      q: 'za-0',
      w: 'za-1',
      e: 'za-2',
      r: 'za-3'
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      const padId = keyMap[e.key.toLowerCase()]
      if (padId) {
        const pad = pads.find((p) => p.id === padId)
        if (pad) triggerPad(pad)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [pads, triggerPad])

  const factorGroups = {
    zc: { color: '#e85d4e', icon: Music, label: isZh ? '内容' : 'Content' },
    zs: { color: '#2d8a5e', icon: Globe, label: isZh ? '文化' : 'Culture' },
    za: { color: '#4a90d9', icon: Heart, label: isZh ? '情感' : 'Affect' }
  }

  return (
    <div className={`flex flex-col overflow-hidden rounded-3xl border border-ink/15 bg-white backdrop-blur-sm ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-ink/15 px-3 py-2">
        <div className="flex items-center gap-2">
          <Volume2 size={14} className="text-textSub" />
          <span className="text-xs font-semibold text-textMain">
            {isZh ? '因子音垫' : 'Factor Pads'}
          </span>
        </div>
        <span className="rounded-full bg-ink/10 px-2 py-0.5 text-[10px] text-textSub">
          {audioEnabled ? (isZh ? '音频就绪' : 'Audio Ready') : (isZh ? '点击激活' : 'Click to Activate')}
        </span>
      </div>

      {/* Pads area - 使用 flex 布局 */}
      <div className="flex flex-col flex-1 p-3 gap-2">
        {/* Factor labels row */}
        <div className="flex justify-around">
          {Object.entries(factorGroups).map(([factor, config]) => {
            const Icon = config.icon
            return (
              <div
                key={factor}
                className="flex items-center gap-1 text-[10px] font-medium"
                style={{ color: config.color }}
              >
                <Icon size={10} />
                {config.label}
              </div>
            )
          })}
        </div>

        {/* Pads grid - 3行4列，更大尺寸 */}
        <div className="grid grid-cols-4 gap-2 flex-1">
          {pads.map((pad) => {
            const isActive = activePads.has(pad.id)
            const factorConfig = factorGroups[pad.factor]

            return (
              <motion.button
                key={pad.id}
                className="flex items-center justify-center w-full h-full"
                onClick={() => triggerPad(pad)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="flex flex-col items-center justify-center rounded-xl border w-full h-full min-h-[56px]"
                  style={{
                    borderColor: isActive ? factorConfig.color : `${factorConfig.color}33`,
                    backgroundColor: isActive ? `${factorConfig.color}33` : `${factorConfig.color}11`,
                    boxShadow: isActive ? `0 0 20px ${factorConfig.color}44` : 'none'
                  }}
                  animate={isActive ? { scale: [1, 1.05, 1] } : {}}
                >
                  <span
                    className="font-mono text-base font-bold"
                    style={{ color: factorConfig.color }}
                  >
                    {pad.note}
                    <sup className="text-[9px]">{pad.octave}</sup>
                  </span>
                  <span className="text-[8px] text-textSub">{pad.factor}</span>
                </motion.div>

                {/* Ripple effect */}
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      className="pointer-events-none absolute inset-0 rounded-xl"
                      style={{ borderColor: factorConfig.color, borderWidth: 2 }}
                      initial={{ scale: 1, opacity: 1 }}
                      animate={{ scale: 1.5, opacity: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.4 }}
                    />
                  )}
                </AnimatePresence>
              </motion.button>
            )
          })}
        </div>
      </div>

      {/* Instructions */}
      <div className="border-t border-ink/15 px-4 py-2">
        <p className="text-[10px] text-textSub">
          {isZh
            ? '按键: A S D F (内容) · J K L ; (文化) · Q W E R (情感)'
            : 'Keys: A S D F (Content) · J K L ; (Culture) · Q W E R (Affect)'}
        </p>
      </div>
    </div>
  )
}
