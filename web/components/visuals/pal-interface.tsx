'use client'

import { useMemo, useState } from 'react'
import { Plus, Send } from 'lucide-react'

import { palUncertaintyGrid } from '@/data/mock-data'
import { clamp, cn } from '@/lib/utils'

type Annotation = {
  sampleId: string
  affectLabel: string
  cultureLabel: string
  rationale: string
}

const affectOptions = ['Calm', 'Joyful', 'Melancholic', 'Mystic', 'Ritual']

export function PalInterface() {
  const [selectedId, setSelectedId] = useState<string>(palUncertaintyGrid[0].id)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [newConcept, setNewConcept] = useState('')
  const [concepts, setConcepts] = useState<string[]>(['Raga', 'Maqam', 'Han', 'Saudade'])

  const selected = useMemo(() => palUncertaintyGrid.find((item) => item.id === selectedId) ?? palUncertaintyGrid[0], [selectedId])

  const [affectLabel, setAffectLabel] = useState('Mystic')
  const [cultureLabel, setCultureLabel] = useState<string>(selected.culture)
  const [rationale, setRationale] = useState('')

  const coverage = useMemo(() => {
    const base = [94, 76, 62, 58, 69, 57, 51, 65]
    const gain = Math.min(12, annotations.length * 0.8 + concepts.length * 0.2)
    return base.map((value) => clamp((value + gain) / 100, 0, 1))
  }, [annotations.length, concepts.length])

  const submitAnnotation = () => {
    const payload: Annotation = {
      sampleId: selected.id,
      affectLabel,
      cultureLabel,
      rationale
    }
    setAnnotations((prev) => [payload, ...prev].slice(0, 10))
    setRationale('')
  }

  const addConcept = () => {
    const cleaned = newConcept.trim()
    if (!cleaned) return
    if (concepts.includes(cleaned)) return
    setConcepts((prev) => [cleaned, ...prev])
    setNewConcept('')
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
      <div className="rounded-3xl paper-card p-5">
        <span className="chapter-chip">uncertainty map</span>
        <h3 className="mt-2 font-display text-3xl text-textMain">Expert Attention Heatmap</h3>
        <p className="mb-4 text-sm text-textSub">Darker cells indicate samples where model confidence is weak and expert intervention has high value.</p>

        <div className="grid grid-cols-8 gap-2">
          {palUncertaintyGrid.map((cell) => {
            const intensity = clamp(cell.value, 0, 1)
            const selectedCell = selected.id === cell.id
            return (
              <button
                key={cell.id}
                onClick={() => {
                  setSelectedId(cell.id)
                  setCultureLabel(cell.culture)
                }}
                className={cn('aspect-square rounded-md border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs', selectedCell ? 'border-zs shadow-[0_6px_16px_rgba(0,167,160,0.28)]' : 'border-ink/15')}
                style={{ background: `rgba(126,87,194,${0.12 + intensity * 0.82})` }}
                aria-label={`${cell.hint} uncertainty ${Math.round(cell.value * 100)} percent`}
              />
            )
          })}
        </div>

        <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-4">
          <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-textSub">feedback loop dynamics</p>
          <p className="text-xs text-textSub">Edges tighten as annotations reduce uncertainty in local ontology neighborhoods.</p>
          <svg viewBox="0 0 400 140" className="mt-3 h-36 w-full rounded-xl border border-ink/15 bg-[#fff8ee]">
            {Array.from({ length: 6 }).map((_, i) => {
              const x1 = 30 + i * 58
              const y1 = 40 + ((i % 2) * 32)
              const x2 = x1 + 45
              const y2 = 88 - ((i % 2) * 28)
              const confidence = clamp((annotations.length + i * 0.8) / 12, 0.1, 1)
              return <line key={`edge-${i}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(0,167,160,0.85)" strokeWidth={1 + confidence * 3} strokeDasharray="6 6" />
            })}
          </svg>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-3xl paper-card p-5">
          <span className="chapter-chip">annotation panel</span>
          <h3 className="mt-2 font-display text-2xl text-textMain">Expert Annotation</h3>
          <p className="text-sm text-textSub">Sample: {selected.hint} · {selected.culture} · uncertainty {(selected.value * 100).toFixed(1)}%</p>

          <div className="mt-4 space-y-3">
            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Affect Label</span>
              <select value={affectLabel} onChange={(event) => setAffectLabel(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain">
                {affectOptions.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Culture Label</span>
              <input value={cultureLabel} onChange={(event) => setCultureLabel(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Free-text rationale</span>
              <textarea rows={3} value={rationale} onChange={(event) => setRationale(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" placeholder="Explain cultural context, performance practice, or affect semantics..." />
            </label>

            <button onClick={submitAnnotation} className="inline-flex items-center rounded-full bg-zs px-4 py-2 text-sm font-semibold text-white">
              <Send size={14} className="mr-2" />Submit Annotation
            </button>
          </div>

          <div className="mt-4 space-y-2 text-xs text-textSub">
            {annotations.length === 0 ? <p>No annotation submitted yet.</p> : null}
            {annotations.map((item, index) => (
              <div key={`${item.sampleId}-${index}`} className="rounded-lg border border-ink/15 bg-white px-2 py-1.5">
                {item.sampleId} · {item.cultureLabel} · {item.affectLabel}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl paper-card p-5">
          <span className="chapter-chip">ontology expansion</span>
          <h4 className="mt-2 font-display text-xl text-textMain">Concept Builder</h4>
          <div className="mt-2 flex gap-2">
            <input value={newConcept} onChange={(event) => setNewConcept(event.target.value)} placeholder="Add concept node (e.g. Han, Saudade)" className="flex-1 rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" />
            <button onClick={addConcept} className="rounded-xl border border-zc/50 bg-zc/10 px-3 py-2 text-zc">
              <Plus size={16} />
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {concepts.map((concept) => (
              <span key={concept} className="rounded-full border border-ink/20 bg-white px-2 py-1 text-xs text-textSub">{concept}</span>
            ))}
          </div>

          <div className="mt-4">
            <p className="mb-2 text-xs text-textSub">Cognitive Justice Indicator (ontology coverage)</p>
            <div className="space-y-2">
              {coverage.map((value, index) => (
                <div key={`cov-${index}`}>
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>Culture #{index + 1}</span>
                    <span>{(value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-gradient-to-r from-zs to-za" style={{ width: `${value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}