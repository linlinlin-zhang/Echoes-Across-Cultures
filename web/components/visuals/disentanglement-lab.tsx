'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'

import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp, cn } from '@/lib/utils'

const presets = [
  { id: 'raga', zhName: '印度拉格动机', enName: 'Indian Raga Motif', notes: ['D4', 'E4', 'G4', 'A4', 'C5'] },
  { id: 'maqam', zhName: '阿拉伯玛卡姆片段', enName: 'Arabic Maqam Gesture', notes: ['C4', 'Db4', 'F4', 'G4', 'Bb4'] },
  { id: 'guqin', zhName: '古琴静修乐句', enName: 'Guqin Meditation Phrase', notes: ['G3', 'A3', 'C4', 'D4', 'G4'] }
]

function useSpectrumCanvas(canvasRef: React.RefObject<HTMLCanvasElement>, spectrum: number[], color: string) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = 'rgba(255,250,241,.95)'
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
  const [statusText, setStatusText] = useState('')
  const [uploadedName, setUploadedName] = useState('')
  const [snapshot, setSnapshot] = useState<{ zc: number; zs: number; za: number } | null>(null)
  const [morphing, setMorphing] = useState(false)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const synthRef = useRef<Tone.PolySynth | null>(null)
  const analyserRef = useRef<Tone.Analyser | null>(null)
  const playbackTimerRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)
  const morphTimerRef = useRef<number | null>(null)

  const [processedSpectrum, setProcessedSpectrum] = useState<number[]>(Array.from({ length: 48 }, () => 0.1))
  const [originalSpectrum, setOriginalSpectrum] = useState<number[]>(Array.from({ length: 48 }, () => 0.08))

  const processedCanvasRef = useRef<HTMLCanvasElement>(null)
  const originalCanvasRef = useRef<HTMLCanvasElement>(null)

  useSpectrumCanvas(processedCanvasRef, processedSpectrum, '#188038')
  useSpectrumCanvas(originalCanvasRef, originalSpectrum, '#ea4335')

  const selectedPreset = useMemo(() => presets.find((item) => item.id === presetId) ?? presets[0], [presetId])
  const selectedPresetLabel = isZh ? `${selectedPreset.zhName}（${selectedPreset.enName}）` : selectedPreset.enName

  const valence = useMemo(() => Number((za * 2 - 1).toFixed(2)), [za])
  const arousal = useMemo(() => Number((zs * 1.6 - 0.8).toFixed(2)), [zs])
  const cosineSimilarityToWestern = useMemo(() => Number((0.2 + (1 - Math.abs(zs - 0.35)) * 0.72).toFixed(2)), [zs])

  const pianoPattern = useMemo(() => {
    const density = Math.round(5 + zc * 8)
    return Array.from({ length: density }).map((_, index) => {
      const x = (index / density) * 100
      const y = 12 + ((index * 13 + Math.round(zc * 10)) % 68)
      const w = 5 + zc * 7
      return { x, y, w }
    })
  }, [zc])

  const snapshotDelta = useMemo(() => {
    if (!snapshot) return null
    return {
      zc: zc - snapshot.zc,
      zs: zs - snapshot.zs,
      za: za - snapshot.za
    }
  }, [snapshot, zc, zs, za])

  const syncAnalyser = useCallback(() => {
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
  }, [zc])

  const stopPattern = useCallback(() => {
    setIsPlaying(false)
    if (playbackTimerRef.current) {
      window.clearInterval(playbackTimerRef.current)
      playbackTimerRef.current = null
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    setStatusText(isZh ? '播放已停止（Playback Stopped）。调整滑杆后可继续试听。' : 'Playback stopped. Adjust sliders and replay.')
  }, [isZh])

  const initializeAudio = useCallback(async () => {
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
    setStatusText(isZh ? '音频上下文已就绪（Audio Context Ready），点击播放试听因子控制迁移。' : 'Audio context ready. Press Play to hear factor-controlled transfer.')
  }, [audioReady, isZh])

  const playPattern = useCallback(async () => {
    if (!audioReady) await initializeAudio()
    if (!synthRef.current) return

    setIsPlaying(true)
    setStatusText(
      isZh
        ? `播放中：${selectedPresetLabel}；参数 zc=${zc.toFixed(2)}，zs=${zs.toFixed(2)}，za=${za.toFixed(2)}`
        : `Playing ${selectedPreset.enName} with zc=${zc.toFixed(2)}, zs=${zs.toFixed(2)}, za=${za.toFixed(2)}`
    )

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
    if (playbackTimerRef.current) window.clearInterval(playbackTimerRef.current)
    playbackTimerRef.current = window.setInterval(trigger, Math.max(140, 520 - zs * 260))
    if (!rafRef.current) syncAnalyser()
  }, [audioReady, initializeAudio, isZh, selectedPreset.enName, selectedPreset.notes, selectedPresetLabel, syncAnalyser, za, zc, zs])

  const applySurprise = () => {
    const nzc = Number((0.35 + Math.random() * 0.6).toFixed(2))
    const nzs = Number((0.2 + Math.random() * 0.75).toFixed(2))
    const nza = Number((0.25 + Math.random() * 0.7).toFixed(2))
    setZc(nzc)
    setZs(nzs)
    setZa(nza)
    setStatusText(
      isZh
        ? `已生成随机组合（Surprise Set）：zc=${nzc.toFixed(2)}，zs=${nzs.toFixed(2)}，za=${nza.toFixed(2)}`
        : `Surprise set applied: zc=${nzc.toFixed(2)}, zs=${nzs.toFixed(2)}, za=${nza.toFixed(2)}`
    )
  }

  const startMorph = () => {
    if (morphTimerRef.current) window.clearInterval(morphTimerRef.current)
    const target = {
      zc: Number((0.3 + Math.random() * 0.65).toFixed(2)),
      zs: Number((0.2 + Math.random() * 0.75).toFixed(2)),
      za: Number((0.2 + Math.random() * 0.75).toFixed(2))
    }

    let frame = 0
    const total = 22
    const start = { zc, zs, za }
    setMorphing(true)
    setStatusText(isZh ? `平滑迁移中（Morphing） -> zc=${target.zc}, zs=${target.zs}, za=${target.za}` : `Morphing -> zc=${target.zc}, zs=${target.zs}, za=${target.za}`)

    morphTimerRef.current = window.setInterval(() => {
      frame += 1
      const alpha = clamp(frame / total, 0, 1)
      setZc(Number((start.zc + (target.zc - start.zc) * alpha).toFixed(2)))
      setZs(Number((start.zs + (target.zs - start.zs) * alpha).toFixed(2)))
      setZa(Number((start.za + (target.za - start.za) * alpha).toFixed(2)))

      if (alpha >= 1 && morphTimerRef.current) {
        window.clearInterval(morphTimerRef.current)
        morphTimerRef.current = null
        setMorphing(false)
      }
    }, 55)
  }

  useEffect(() => {
    setStatusText(isZh ? '音频上下文待激活（Audio Context is Idle）。点击“初始化”开始。' : 'Audio context is idle. Click Initialize to start.')
  }, [isZh])

  useEffect(() => {
    return () => {
      if (morphTimerRef.current) window.clearInterval(morphTimerRef.current)
      stopPattern()
      synthRef.current?.dispose()
      analyserRef.current?.dispose()
    }
  }, [stopPattern])

  return (
    <div className="grid gap-6 xl:grid-cols-[0.96fr_1.04fr]">
      <div className="space-y-4 rounded-3xl paper-card p-5">
        <div>
          <span className="chapter-chip">{isZh ? '章节工作台（Chapter Worksheet）' : 'chapter worksheet'}</span>
          <h3 className="mt-2 font-display text-3xl text-textMain">{isZh ? '风格迁移工作台（Style Transfer Workbench）' : 'Style Transfer Workbench'}</h3>
          <p className="mt-2 text-sm text-textSub">{isZh ? '依次完成源样本选择、潜变量调节与实时解释视图观察。' : 'Step through source selection, latent tuning, and realtime interpretation views.'}</p>
        </div>

        <div className="grid gap-3">
          <div className="note-card">
            <p className="font-mono text-[11px] uppercase tracking-[0.13em] text-textSub">{isZh ? '步骤 1 · 源样本（Step 1 · Source）' : 'step 1 · source'}</p>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs text-textSub">{isZh ? '预设样本（Preset Sample）' : 'Preset Sample'}</span>
                <select value={presetId} onChange={(event) => setPresetId(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain">
                  {presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {isZh ? `${preset.zhName}（${preset.enName}）` : preset.enName}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-xs text-textSub">{isZh ? '上传音频（占位演示）' : 'Upload Audio (placeholder)'}</span>
                <input type="file" accept="audio/*" onChange={(event) => setUploadedName(event.target.files?.[0]?.name ?? '')} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-xs text-textMain" />
              </label>
            </div>
            {uploadedName ? <p className="mt-2 font-mono text-xs text-textSub">{isZh ? `已载入（Loaded）：${uploadedName}` : `Loaded: ${uploadedName}`}</p> : null}
          </div>

          <div className="note-card">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.13em] text-textSub">{isZh ? '步骤 2 · 潜变量调节（Step 2 · Latent Tuning）' : 'step 2 · latent tuning'}</p>
              <div className="flex flex-wrap gap-1.5">
                <button onClick={applySurprise} className="rounded-full border border-ink/20 bg-white px-2.5 py-1 text-xs text-textSub hover:text-textMain">
                  {isZh ? '随机探索（Surprise）' : 'Surprise'}
                </button>
                <button
                  onClick={startMorph}
                  disabled={morphing}
                  className={cn('rounded-full border px-2.5 py-1 text-xs', morphing ? 'border-za/30 bg-za/10 text-za' : 'border-ink/20 bg-white text-textSub hover:text-textMain')}
                >
                  {isZh ? (morphing ? '迁移中（Morphing）' : '平滑迁移（Morph）') : morphing ? 'Morphing' : 'Morph'}
                </button>
                <button
                  onClick={() => setSnapshot({ zc, zs, za })}
                  className="rounded-full border border-ink/20 bg-white px-2.5 py-1 text-xs text-textSub hover:text-textMain"
                >
                  {isZh ? '记录快照（Snapshot）' : 'Snapshot'}
                </button>
              </div>
            </div>

            <div className="mt-2 space-y-3">
              {[
                { key: 'zc', label: isZh ? 'zc - 内容保留（Content Retention）' : 'zc - Content Retention', value: zc, set: setZc, color: 'bg-zc' },
                { key: 'zs', label: isZh ? 'zs - 风格强度（Style Strength）' : 'zs - Style Strength', value: zs, set: setZs, color: 'bg-zs' },
                { key: 'za', label: isZh ? 'za - 情感保留（Affect Preservation）' : 'za - Affect Preservation', value: za, set: setZa, color: 'bg-za' }
              ].map((item) => (
                <label key={item.key} className="block">
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{item.label}</span>
                    <span>{item.value.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.01} value={item.value} onChange={(event) => item.set(Number(event.target.value))} className="w-full accent-zs" />
                  <div className="mt-1 h-1 rounded-full bg-ink/10">
                    <div className={`h-full rounded-full ${item.color}`} style={{ width: `${item.value * 100}%` }} />
                  </div>
                </label>
              ))}
            </div>

            {snapshotDelta ? (
              <div className="mt-3 rounded-xl border border-ink/15 bg-white px-3 py-2 text-xs text-textSub">
                {isZh ? '快照差值（Snapshot Delta）：' : 'Snapshot Delta:'} zc {snapshotDelta.zc >= 0 ? '+' : ''}
                {snapshotDelta.zc.toFixed(2)} · zs {snapshotDelta.zs >= 0 ? '+' : ''}
                {snapshotDelta.zs.toFixed(2)} · za {snapshotDelta.za >= 0 ? '+' : ''}
                {snapshotDelta.za.toFixed(2)}
              </div>
            ) : null}
          </div>

          <div className="note-card">
            <p className="font-mono text-[11px] uppercase tracking-[0.13em] text-textSub">{isZh ? '步骤 3 · 试听（Step 3 · Audition）' : 'step 3 · audition'}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <button onClick={initializeAudio} className="rounded-full border border-zs/50 bg-zs/10 px-4 py-2 text-sm font-semibold text-zs">
                {isZh ? '初始化（Initialize）' : 'Initialize'}
              </button>
              <button onClick={playPattern} disabled={!audioReady && isPlaying} className="rounded-full bg-zc px-4 py-2 text-sm font-semibold text-white">
                {isZh ? '播放（Play）' : 'Play'}
              </button>
              <button onClick={stopPattern} className="rounded-full border border-ink/25 bg-white px-4 py-2 text-sm font-semibold text-textMain">
                {isZh ? '停止（Stop）' : 'Stop'}
              </button>
            </div>
            <div className="mt-3 rounded-xl border border-ink/15 bg-white p-3 font-mono text-xs text-textSub">{statusText}</div>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="paper-card rounded-2xl p-3">
            <h4 className="font-display text-base text-textMain">{isZh ? '原始频谱（Original Spectrum）' : 'Original Spectrum'}</h4>
            <canvas ref={originalCanvasRef} width={420} height={180} className="mt-2 h-44 w-full rounded-lg border border-ink/15" />
          </div>
          <div className="paper-card rounded-2xl p-3">
            <h4 className="font-display text-base text-textMain">{isZh ? '迁移后频谱（Transferred Spectrum）' : 'Transferred Spectrum'}</h4>
            <canvas ref={processedCanvasRef} width={420} height={180} className="mt-2 h-44 w-full rounded-lg border border-ink/15" />
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="paper-card rounded-2xl p-4">
            <h4 className="font-display text-base text-textMain">{isZh ? '内容表征（Content Representation）' : 'Content Representation'}</h4>
            <p className="mb-3 text-xs text-textSub">{isZh ? '钢琴卷帘轮廓近似（Piano-roll Contour Approximation, zc）' : 'Piano-roll contour approximation (zc)'}</p>
            <svg viewBox="0 0 100 80" className="h-28 w-full rounded-lg bg-[#fff8ee]">
              {Array.from({ length: 8 }).map((_, i) => (
                <line key={`h-${i}`} x1={0} y1={i * 10} x2={100} y2={i * 10} stroke="rgba(84,95,120,0.2)" strokeWidth={0.4} />
              ))}
              {pianoPattern.map((note, idx) => (
                <rect key={`n-${idx}`} x={note.x} y={note.y} width={note.w} height={5.5} fill="#ea4335" opacity={0.86} rx={1.2} />
              ))}
            </svg>
          </div>

          <div className="paper-card rounded-2xl p-4">
            <h4 className="font-display text-base text-textMain">{isZh ? '情感平面（Affect Plane）' : 'Affect Plane'}</h4>
            <p className="mb-3 text-xs text-textSub">{isZh ? '效价-唤醒轨迹（Valence-Arousal Trajectory, za）' : 'Valence-Arousal trajectory (za)'}</p>
            <div className="relative h-28 rounded-lg border border-ink/15 bg-[#fff8ee]">
              <div className="absolute left-1/2 top-0 h-full w-px bg-ink/20" />
              <div className="absolute left-0 top-1/2 h-px w-full bg-ink/20" />
              <div className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-za shadow-[0_0_12px_rgba(26,115,232,0.6)]" style={{ left: `${(valence + 1) * 50}%`, top: `${(1 - (arousal + 1) / 2) * 100}%` }} />
            </div>
            <p className="mt-2 font-mono text-xs text-textSub">V={valence.toFixed(2)} · A={arousal.toFixed(2)}</p>
          </div>

          <div className="paper-card rounded-2xl p-4">
            <h4 className="font-display text-base text-textMain">{isZh ? '文化相似度（Culture Similarity）' : 'Culture Similarity'}</h4>
            <p className="mb-3 text-xs text-textSub">{isZh ? '与西方锚点余弦相似度（Cosine Similarity vs Western Anchor）' : 'Cosine similarity vs Western anchor'}</p>
            <div className="relative mx-auto h-28 w-28">
              <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
                <circle cx="60" cy="60" r="44" fill="none" stroke="rgba(84,95,120,0.2)" strokeWidth="10" />
                <circle cx="60" cy="60" r="44" fill="none" stroke="#188038" strokeWidth="10" strokeLinecap="round" strokeDasharray={`${Math.PI * 2 * 44 * cosineSimilarityToWestern} ${Math.PI * 2 * 44}`} />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center font-mono text-sm text-textMain">{(cosineSimilarityToWestern * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

