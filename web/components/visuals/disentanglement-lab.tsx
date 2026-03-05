'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'

import { clamp } from '@/lib/utils'

const presets = [
  { id: 'raga', name: 'Indian Raga Motif', notes: ['D4', 'E4', 'G4', 'A4', 'C5'] },
  { id: 'maqam', name: 'Arabic Maqam Gesture', notes: ['C4', 'Db4', 'F4', 'G4', 'Bb4'] },
  { id: 'guqin', name: 'Guqin Meditation Phrase', notes: ['G3', 'A3', 'C4', 'D4', 'G4'] }
]

function useSpectrumCanvas(canvasRef: React.RefObject<HTMLCanvasElement>, spectrum: number[], color: string) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = 'rgba(8,12,24,.7)'
    ctx.fillRect(0, 0, width, height)

    const barWidth = width / spectrum.length
    spectrum.forEach((value, index) => {
      const normalized = clamp(Math.abs(value), 0, 1)
      const h = normalized * height
      ctx.fillStyle = color
      ctx.fillRect(index * barWidth, height - h, Math.max(2, barWidth - 2), h)
    })
  }, [canvasRef, spectrum, color])
}

export function DisentanglementLab() {
  const [zc, setZc] = useState(0.85)
  const [zs, setZs] = useState(0.65)
  const [za, setZa] = useState(0.72)
  const [presetId, setPresetId] = useState(presets[0].id)
  const [audioReady, setAudioReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [statusText, setStatusText] = useState('Audio context is idle. Click "Initialize" to start.')
  const [uploadedName, setUploadedName] = useState('')

  const synthRef = useRef<Tone.PolySynth | null>(null)
  const analyserRef = useRef<Tone.Analyser | null>(null)
  const playbackTimerRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)

  const [processedSpectrum, setProcessedSpectrum] = useState<number[]>(Array.from({ length: 48 }, () => 0.1))
  const [originalSpectrum, setOriginalSpectrum] = useState<number[]>(Array.from({ length: 48 }, () => 0.08))

  const processedCanvasRef = useRef<HTMLCanvasElement>(null)
  const originalCanvasRef = useRef<HTMLCanvasElement>(null)

  useSpectrumCanvas(processedCanvasRef, processedSpectrum, '#4ecdc4')
  useSpectrumCanvas(originalCanvasRef, originalSpectrum, '#ff6b6b')

  const selectedPreset = useMemo(() => presets.find((item) => item.id === presetId) ?? presets[0], [presetId])

  const valence = useMemo(() => Number((za * 2 - 1).toFixed(2)), [za])
  const arousal = useMemo(() => Number((zs * 1.6 - 0.8).toFixed(2)), [zs])
  const cosineSimilarityToWestern = useMemo(() => Number((0.2 + (1 - Math.abs(zs - 0.35)) * 0.72).toFixed(2)), [zs])

  const pianoPattern = useMemo(() => {
    const density = Math.round(5 + zc * 8)
    return Array.from({ length: density }).map((_, index) => {
      const x = (index / density) * 100
      const y = 12 + ((index * 13 + Math.round(zc * 10)) % 68)
      const w = 5 + (zc * 7)
      return { x, y, w }
    })
  }, [zc])

  const syncAnalyser = () => {
    if (!analyserRef.current) return
    const raw = analyserRef.current.getValue() as Float32Array
    const normalized = Array.from(raw.slice(0, 48)).map((value) => clamp((value + 120) / 100, 0, 1))
    setProcessedSpectrum(normalized)
    setOriginalSpectrum((prev) =>
      prev.map((_, idx) => {
        const t = performance.now() * 0.001
        return clamp(0.15 + Math.sin(t * 1.5 + idx * 0.3 + zc) * 0.2 + (1 - zc) * 0.15, 0, 1)
      })
    )
    rafRef.current = requestAnimationFrame(syncAnalyser)
  }

  const initializeAudio = async () => {
    if (audioReady) return
    await Tone.start()
    const analyser = new Tone.Analyser('fft', 128)
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'fatsawtooth' },
      envelope: { attack: 0.04, decay: 0.22, sustain: 0.35, release: 0.6 }
    })
    synth.connect(analyser)
    synth.toDestination()
    analyserRef.current = analyser
    synthRef.current = synth
    setAudioReady(true)
    setStatusText('Audio context ready. Press Play to hear factor-controlled style transfer demo.')
  }

  const playPattern = async () => {
    if (!audioReady) {
      await initializeAudio()
    }
    if (!synthRef.current) return

    setIsPlaying(true)
    setStatusText(`Playing ${selectedPreset.name} with zc=${zc.toFixed(2)}, zs=${zs.toFixed(2)}, za=${za.toFixed(2)}`)

    let step = 0
    const noteLength = Math.max(0.15, 0.42 - zc * 0.2)

    const trigger = () => {
      if (!synthRef.current) return
      const note = selectedPreset.notes[step % selectedPreset.notes.length]
      const shifted = Tone.Frequency(note).transpose(Math.round((zs - 0.5) * 7)).toNote()
      synthRef.current.volume.value = -8 + za * 8
      synthRef.current.triggerAttackRelease(shifted, noteLength)
      step += 1
    }

    trigger()
    if (playbackTimerRef.current) {
      window.clearInterval(playbackTimerRef.current)
    }
    playbackTimerRef.current = window.setInterval(trigger, Math.max(140, 520 - zs * 260))

    if (!rafRef.current) {
      syncAnalyser()
    }
  }

  const stopPattern = () => {
    setIsPlaying(false)
    if (playbackTimerRef.current) {
      window.clearInterval(playbackTimerRef.current)
      playbackTimerRef.current = null
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    setStatusText('Playback stopped. Adjust latent sliders and play again.')
  }

  useEffect(() => {
    return () => {
      stopPattern()
      synthRef.current?.dispose()
      analyserRef.current?.dispose()
    }
  }, [])

  return (
    <div className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
      <div className="space-y-4 rounded-3xl border border-white/10 bg-black/30 p-5">
        <h3 className="font-display text-2xl text-textMain">Style Transfer Control Panel</h3>
        <p className="text-sm text-textSub">Independent latent sliders control content retention, culture style strength, and affect preservation.</p>

        <div className="space-y-3">
          {[
            { key: 'zc', label: 'zc - Content Retention', value: zc, set: setZc, color: 'bg-zc' },
            { key: 'zs', label: 'zs - Style Strength', value: zs, set: setZs, color: 'bg-zs' },
            { key: 'za', label: 'za - Affect Preservation', value: za, set: setZa, color: 'bg-za' }
          ].map((item) => (
            <label key={item.key} className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>{item.label}</span>
                <span>{item.value.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={item.value}
                onChange={(event) => item.set(Number(event.target.value))}
                className="w-full accent-zs"
              />
              <div className="mt-1 h-1 rounded-full bg-white/10">
                <div className={`h-full rounded-full ${item.color}`} style={{ width: `${item.value * 100}%` }} />
              </div>
            </label>
          ))}
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-xs text-textSub">Preset Sample</span>
            <select
              value={presetId}
              onChange={(event) => setPresetId(event.target.value)}
              className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-xs text-textSub">Upload Audio (placeholder)</span>
            <input
              type="file"
              accept="audio/*"
              onChange={(event) => setUploadedName(event.target.files?.[0]?.name ?? '')}
              className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-xs text-textMain"
            />
          </label>
        </div>

        {uploadedName ? <p className="font-mono text-xs text-textSub">Loaded: {uploadedName}</p> : null}

        <div className="flex flex-wrap gap-2">
          <button
            onClick={initializeAudio}
            className="rounded-full border border-zs/50 bg-zs/10 px-4 py-2 text-sm font-semibold text-zs"
          >
            Initialize Audio
          </button>
          <button
            onClick={playPattern}
            disabled={!audioReady && isPlaying}
            className="rounded-full bg-zc px-4 py-2 text-sm font-semibold text-abyss"
          >
            Play Transfer Demo
          </button>
          <button
            onClick={stopPattern}
            className="rounded-full border border-white/25 bg-white/5 px-4 py-2 text-sm font-semibold text-textMain"
          >
            Stop
          </button>
        </div>

        <div className="rounded-xl border border-white/15 bg-black/40 p-3 font-mono text-xs text-textSub">{statusText}</div>
      </div>

      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-black/25 p-3">
            <h4 className="mb-2 font-display text-base text-textMain">Original Spectrum</h4>
            <canvas ref={originalCanvasRef} width={420} height={180} className="h-44 w-full rounded-lg border border-white/10" />
          </div>
          <div className="rounded-2xl border border-white/10 bg-black/25 p-3">
            <h4 className="mb-2 font-display text-base text-textMain">Transferred Spectrum</h4>
            <canvas ref={processedCanvasRef} width={420} height={180} className="h-44 w-full rounded-lg border border-white/10" />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-black/25 p-4">
            <h4 className="font-display text-base text-textMain">Content Representation</h4>
            <p className="mb-3 text-xs text-textSub">Piano-roll contour approximation (zc)</p>
            <svg viewBox="0 0 100 80" className="h-28 w-full rounded-lg bg-black/30">
              {Array.from({ length: 8 }).map((_, i) => (
                <line key={`h-${i}`} x1={0} y1={i * 10} x2={100} y2={i * 10} stroke="rgba(148,163,184,0.25)" strokeWidth={0.4} />
              ))}
              {pianoPattern.map((note, idx) => (
                <rect key={`n-${idx}`} x={note.x} y={note.y} width={note.w} height={5.5} fill="#ff6b6b" opacity={0.86} rx={1.2} />
              ))}
            </svg>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/25 p-4">
            <h4 className="font-display text-base text-textMain">Affect Plane</h4>
            <p className="mb-3 text-xs text-textSub">Valence-Arousal trajectory (za)</p>
            <div className="relative h-28 rounded-lg border border-white/10 bg-black/30">
              <div className="absolute left-1/2 top-0 h-full w-px bg-white/20" />
              <div className="absolute left-0 top-1/2 h-px w-full bg-white/20" />
              <div
                className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-za shadow-[0_0_12px_rgba(165,94,234,0.7)]"
                style={{ left: `${(valence + 1) * 50}%`, top: `${(1 - (arousal + 1) / 2) * 100}%` }}
              />
            </div>
            <p className="mt-2 font-mono text-xs text-textSub">V={valence.toFixed(2)} · A={arousal.toFixed(2)}</p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/25 p-4">
            <h4 className="font-display text-base text-textMain">Culture Vector Similarity</h4>
            <p className="mb-3 text-xs text-textSub">Cosine similarity vs Western anchor</p>
            <div className="relative mx-auto h-28 w-28">
              <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
                <circle cx="60" cy="60" r="44" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="10" />
                <circle
                  cx="60"
                  cy="60"
                  r="44"
                  fill="none"
                  stroke="#4ecdc4"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray={`${Math.PI * 2 * 44 * cosineSimilarityToWestern} ${Math.PI * 2 * 44}`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center font-mono text-sm text-textMain">
                {(cosineSimilarityToWestern * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
