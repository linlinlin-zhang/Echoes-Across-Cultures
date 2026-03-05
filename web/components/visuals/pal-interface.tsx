'use client'

import { useEffect, useMemo, useState } from 'react'
import { BrainCircuit, GaugeCircle, Plus, Send, Shuffle, Sparkles } from 'lucide-react'

import { cultureNodes, palUncertaintyGrid } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cultureLabel, songTitleLabel } from '@/lib/bilingual'
import { clamp, cn } from '@/lib/utils'

type GridCell = (typeof palUncertaintyGrid)[number]

type Annotation = {
  sampleId: string
  affectLabel: string
  cultureLabel: string
  rationale: string
  quality: number
  round: number
}

type AffectOption = {
  en: string
  zh: string
}

type RoundMetric = {
  round: number
  uncertainty: number
  coverage: number
  quality: number
}

const affectOptions: AffectOption[] = [
  { en: 'Calm', zh: '平静' },
  { en: 'Joyful', zh: '喜悦' },
  { en: 'Melancholic', zh: '忧郁' },
  { en: 'Mystic', zh: '神秘' },
  { en: 'Ritual', zh: '仪式感' }
]

const conceptZhMap: Record<string, string> = {
  Raga: '拉格',
  Maqam: '玛卡姆',
  Han: '汉',
  Saudade: '乡愁'
}

function meanUncertainty(cells: GridCell[]) {
  if (cells.length === 0) return 0
  return cells.reduce((acc, item) => acc + item.value, 0) / cells.length
}

function sparkPoints(metrics: RoundMetric[], key: 'uncertainty' | 'coverage', width: number, height: number) {
  if (metrics.length === 0) return ''
  const pad = 10
  if (metrics.length === 1) {
    const y = pad + (1 - metrics[0][key]) * (height - pad * 2)
    return `${pad},${y} ${width - pad},${y}`
  }

  return metrics
    .map((item, index) => {
      const x = pad + (index / (metrics.length - 1)) * (width - pad * 2)
      const y = pad + (1 - item[key]) * (height - pad * 2)
      return `${x},${y}`
    })
    .join(' ')
}

function normalizeValue(value: string) {
  return value.trim().toLowerCase()
}

export function PalInterface() {
  const initialAverage = useMemo(() => meanUncertainty(palUncertaintyGrid), [])
  const baseCoverage = useMemo(() => cultureNodes.reduce((acc, node) => acc + node.ontologyCoverage, 0) / cultureNodes.length, [])

  const [gridData, setGridData] = useState<GridCell[]>(() => palUncertaintyGrid.map((item) => ({ ...item })))
  const [selectedId, setSelectedId] = useState<string>(palUncertaintyGrid[0].id)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [newConcept, setNewConcept] = useState('')
  const [anchorCulture, setAnchorCulture] = useState<string>('Guqin')
  const [concepts, setConcepts] = useState<string[]>(['Raga', 'Maqam', 'Han', 'Saudade'])
  const [conceptAnchors, setConceptAnchors] = useState<Record<string, string>>({
    Raga: 'Indian Classical',
    Maqam: 'Arabic Maqam',
    Han: 'Guqin',
    Saudade: 'Andean Folk'
  })
  const [affectLabel, setAffectLabel] = useState('Mystic')
  const [cultureInput, setCultureInput] = useState<string>(palUncertaintyGrid[0].culture)
  const [rationale, setRationale] = useState('')
  const [batchSize, setBatchSize] = useState(6)
  const [queue, setQueue] = useState<string[]>([])
  const [queueCursor, setQueueCursor] = useState(0)
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.56)
  const [palRound, setPalRound] = useState(0)
  const [roundMetrics, setRoundMetrics] = useState<RoundMetric[]>([
    {
      round: 0,
      uncertainty: initialAverage,
      coverage: baseCoverage,
      quality: 0.45
    }
  ])

  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const selected = useMemo(() => gridData.find((item) => item.id === selectedId) ?? gridData[0], [gridData, selectedId])

  useEffect(() => {
    setCultureInput(selected.culture)
  }, [selected.id, selected.culture])

  const annotatedIdSet = useMemo(() => new Set(annotations.map((item) => item.sampleId)), [annotations])

  const uncertainHotspots = useMemo(() => gridData.filter((cell) => cell.value >= uncertaintyThreshold), [gridData, uncertaintyThreshold])

  const queueAnnotatedCount = useMemo(() => {
    if (queue.length === 0) return 0
    return queue.filter((id) => annotatedIdSet.has(id)).length
  }, [annotatedIdSet, queue])

  const queueProgress = queue.length === 0 ? 0 : queueAnnotatedCount / queue.length

  const activeLearningGain = useMemo(() => {
    const current = meanUncertainty(gridData)
    return clamp((initialAverage - current) / initialAverage, 0, 1)
  }, [gridData, initialAverage])

  const missionScore = useMemo(() => {
    const textBonus = clamp((annotations[0]?.rationale.length ?? 0) / 60, 0, 1) * 0.35
    const diversityBonus = annotations[0] && normalizeValue(annotations[0].cultureLabel) !== normalizeValue(selected.culture) ? 0.3 : 0.12
    const gainBonus = activeLearningGain * 0.35
    return clamp(textBonus + diversityBonus + gainBonus, 0, 1)
  }, [activeLearningGain, annotations, selected.culture])

  const coverage = useMemo(() => {
    return cultureNodes.map((node) => {
      const nodeName = normalizeValue(node.name)
      const localizedName = normalizeValue(cultureLabel('zh', node.name))

      const annotationBoost =
        annotations.filter((item) => {
          const label = normalizeValue(item.cultureLabel)
          return label.includes(nodeName) || localizedName.includes(label)
        }).length * 0.028

      const conceptBoost = Object.entries(conceptAnchors).filter((entry) => entry[1] === node.name).length * 0.022
      const roundBoost = palRound * 0.011
      const gainBoost = activeLearningGain * 0.09

      return {
        id: node.id,
        label: cultureLabel(locale, node.name),
        value: clamp(node.ontologyCoverage + annotationBoost + conceptBoost + roundBoost + gainBoost, 0, 0.995)
      }
    })
  }, [activeLearningGain, annotations, conceptAnchors, locale, palRound])

  const averageCoverage = useMemo(() => {
    if (coverage.length === 0) return 0
    return coverage.reduce((acc, item) => acc + item.value, 0) / coverage.length
  }, [coverage])

  const uncertaintyPoints = useMemo(() => sparkPoints(roundMetrics, 'uncertainty', 280, 92), [roundMetrics])
  const coveragePoints = useMemo(() => sparkPoints(roundMetrics, 'coverage', 280, 92), [roundMetrics])

  const submitAnnotation = () => {
    const rationaleLength = rationale.trim().length
    const quality = Number((clamp(rationaleLength / 60, 0, 1) * 0.5 + (normalizeValue(cultureInput) === normalizeValue(selected.culture) ? 0.2 : 0.35) + 0.15).toFixed(3))

    const payload: Annotation = {
      sampleId: selected.id,
      affectLabel,
      cultureLabel: cultureInput,
      rationale,
      quality,
      round: palRound
    }

    setAnnotations((prev) => [payload, ...prev].slice(0, 24))

    setGridData((prev) =>
      prev.map((cell) => {
        if (cell.id !== selected.id) return cell
        return {
          ...cell,
          value: Number(clamp(cell.value * 0.76, 0.04, 0.99).toFixed(3))
        }
      })
    )

    setRationale('')
  }

  const addConcept = () => {
    const cleaned = newConcept.trim()
    if (!cleaned) return
    if (concepts.includes(cleaned)) return
    setConcepts((prev) => [cleaned, ...prev])
    setConceptAnchors((prev) => ({ ...prev, [cleaned]: anchorCulture }))
    setNewConcept('')
  }

  const buildBatchQueue = () => {
    const nextQueue = [...gridData]
      .sort((a, b) => b.value - a.value)
      .slice(0, batchSize)
      .map((item) => item.id)

    setQueue(nextQueue)
    setQueueCursor(0)
    if (nextQueue[0]) {
      setSelectedId(nextQueue[0])
    }
  }

  const moveToNextQueued = () => {
    if (queue.length === 0) return
    const nextCursor = (queueCursor + 1) % queue.length
    setQueueCursor(nextCursor)
    setSelectedId(queue[nextCursor])
  }

  const pickRandomHotspot = () => {
    if (uncertainHotspots.length === 0) return
    const randomIndex = Math.floor(Math.random() * uncertainHotspots.length)
    setSelectedId(uncertainHotspots[randomIndex].id)
  }

  const simulatePalRound = () => {
    if (annotations.length === 0) return

    const recent = annotations.slice(0, Math.max(4, batchSize))
    const annotatedIds = new Set(recent.map((item) => item.sampleId))
    const culturesTouched = new Set(recent.map((item) => normalizeValue(item.cultureLabel)))
    const nextRound = palRound + 1

    const nextGrid = gridData.map((cell) => {
      let decay = 0.015 + nextRound * 0.003
      if (annotatedIds.has(cell.id)) decay += 0.22
      if (culturesTouched.has(normalizeValue(cell.culture))) decay += 0.07
      if (cell.value >= uncertaintyThreshold) decay += 0.055

      return {
        ...cell,
        value: Number(clamp(cell.value - decay, 0.035, 0.99).toFixed(3))
      }
    })

    const avgQuality = recent.reduce((acc, item) => acc + item.quality, 0) / recent.length
    const nextCoverage = clamp(baseCoverage + nextRound * 0.02 + concepts.length * 0.006 + avgQuality * 0.08, 0, 0.995)

    setPalRound(nextRound)
    setGridData(nextGrid)
    setRoundMetrics((prev) =>
      [
        ...prev,
        {
          round: nextRound,
          uncertainty: meanUncertainty(nextGrid),
          coverage: nextCoverage,
          quality: avgQuality
        }
      ].slice(-8)
    )
  }

  const renderAffectLabel = (value: string) => {
    const found = affectOptions.find((item) => item.en === value)
    if (!found) return value
    return isZh ? `${found.zh}（${found.en}）` : found.en
  }

  const renderConceptLabel = (value: string) => {
    if (!isZh) return value
    const zh = conceptZhMap[value]
    return zh ? `${zh}（${value}）` : value
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
      <div className="rounded-3xl paper-card p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <span className="chapter-chip">{isZh ? '不确定性地图（Uncertainty Map）' : 'uncertainty map'}</span>
          <span className="sticker">{isZh ? `第 ${palRound} 轮（Round ${palRound}）` : `Round ${palRound}`}</span>
        </div>

        <h3 className="mt-2 font-display text-3xl text-textMain">{isZh ? '专家注意力热力图（Expert Attention Heatmap）' : 'Expert Attention Heatmap'}</h3>
        <p className="mb-3 text-sm text-textSub">
          {isZh
            ? '把 PAL 当成可执行流程：先生成高不确定性批次，再做标注，再模拟约束回灌。'
            : 'Run PAL as an executable loop: build a high-uncertainty batch, annotate, then simulate constrained feedback.'}
        </p>

        <div className="mb-3 grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-end">
          <label className="block">
            <div className="mb-1 flex items-center justify-between text-xs text-textSub">
              <span>{isZh ? '热点阈值（Hotspot Threshold）' : 'Hotspot Threshold'}</span>
              <span>{(uncertaintyThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.3}
              max={0.95}
              step={0.01}
              value={uncertaintyThreshold}
              onChange={(event) => setUncertaintyThreshold(Number(event.target.value))}
              className="w-full accent-zs"
            />
          </label>

          <button onClick={buildBatchQueue} className="inline-flex items-center rounded-full border border-zs/35 bg-zs/10 px-3 py-1.5 text-xs font-semibold text-zs">
            <BrainCircuit size={13} className="mr-1" />
            {isZh ? '生成批次（Build Batch）' : 'Build Batch'}
          </button>

          <button onClick={pickRandomHotspot} className="inline-flex items-center rounded-full border border-ink/20 bg-white px-3 py-1.5 text-xs font-semibold text-textSub hover:text-textMain">
            <Shuffle size={13} className="mr-1" />
            {isZh ? '随机热点（Random Hotspot）' : 'Random Hotspot'}
          </button>
        </div>

        <div className="grid grid-cols-8 gap-2">
          {gridData.map((cell) => {
            const intensity = clamp(cell.value, 0, 1)
            const selectedCell = selected.id === cell.id
            const queued = queue.includes(cell.id)
            const done = annotatedIdSet.has(cell.id)

            return (
              <button
                key={cell.id}
                onClick={() => setSelectedId(cell.id)}
                className={cn(
                  'aspect-square rounded-md border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zs',
                  selectedCell ? 'border-zs shadow-[0_6px_16px_rgba(15,138,133,0.28)]' : 'border-ink/15',
                  queued ? 'ring-1 ring-zc/55 ring-offset-1 ring-offset-white' : '',
                  done ? 'opacity-65' : ''
                )}
                style={{ background: `rgba(53,95,206,${0.12 + intensity * 0.82})` }}
                aria-label={
                  isZh
                    ? `${songTitleLabel(locale, cell.hint)}，不确定性 ${(cell.value * 100).toFixed(0)}%`
                    : `${cell.hint} uncertainty ${Math.round(cell.value * 100)} percent`
                }
              />
            )
          })}
        </div>

        <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-textSub">{isZh ? '主动学习冲刺（Active Learning Sprint）' : 'active learning sprint'}</p>
            <span className="text-xs text-textSub">
              {isZh ? `热点样本 ${uncertainHotspots.length} 个（Hotspots）` : `${uncertainHotspots.length} hotspots`}
            </span>
          </div>

          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <label className="inline-flex items-center gap-2 rounded-full border border-ink/15 bg-[#fff8ee] px-3 py-1.5 text-textSub">
              <span>{isZh ? '批次大小（Batch Size）' : 'Batch Size'}</span>
              <input type="range" min={4} max={12} step={1} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} className="w-16 accent-zc" />
              <span>{batchSize}</span>
            </label>
            <button onClick={moveToNextQueued} className="rounded-full border border-ink/20 bg-white px-3 py-1.5 text-xs font-semibold text-textSub hover:text-textMain">
              {isZh ? '下一个样本（Next）' : 'Next Sample'}
            </button>
            <button onClick={simulatePalRound} className="inline-flex items-center rounded-full bg-zc px-3 py-1.5 text-xs font-semibold text-white">
              <Sparkles size={13} className="mr-1" />
              {isZh ? '模拟回灌（Simulate Feedback）' : 'Simulate Feedback'}
            </button>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {queue.length === 0 ? (
              <p className="text-xs text-textSub">{isZh ? '尚未生成批次。' : 'No batch generated yet.'}</p>
            ) : (
              queue.map((id, index) => {
                const item = gridData.find((cell) => cell.id === id)
                if (!item) return null
                const done = annotatedIdSet.has(id)
                return (
                  <button
                    key={id}
                    onClick={() => {
                      setQueueCursor(index)
                      setSelectedId(id)
                    }}
                    className={cn(
                      'rounded-full border px-2 py-1 text-[11px] transition',
                      queueCursor === index ? 'border-zc/55 bg-zc/10 text-zc' : 'border-ink/15 bg-white text-textSub',
                      done ? 'line-through opacity-60' : ''
                    )}
                  >
                    {songTitleLabel(locale, item.hint)}
                  </button>
                )
              })
            )}
          </div>

          <div className="mt-3 h-1.5 rounded-full bg-ink/10">
            <div className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za" style={{ width: `${(queueProgress * 100).toFixed(1)}%` }} />
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-3xl paper-card p-5">
          <span className="chapter-chip">{isZh ? '标注面板（Annotation Panel）' : 'annotation panel'}</span>
          <h3 className="mt-2 font-display text-2xl text-textMain">{isZh ? '专家标注（Expert Annotation）' : 'Expert Annotation'}</h3>
          <p className="text-sm text-textSub">
            {isZh
              ? `样本：${songTitleLabel(locale, selected.hint)} · ${cultureLabel(locale, selected.culture)} · 不确定性 ${(selected.value * 100).toFixed(1)}%`
              : `Sample: ${selected.hint} · ${selected.culture} · uncertainty ${(selected.value * 100).toFixed(1)}%`}
          </p>

          <div className="mt-4 space-y-3">
            <label className="block">
              <span className="mb-1 block text-xs text-textSub">{isZh ? '情感标签（Affect Label）' : 'Affect Label'}</span>
              <select value={affectLabel} onChange={(event) => setAffectLabel(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain">
                {affectOptions.map((item) => (
                  <option key={item.en} value={item.en}>
                    {isZh ? `${item.zh}（${item.en}）` : item.en}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">{isZh ? '文化标签（Culture Label）' : 'Culture Label'}</span>
              <input value={cultureInput} onChange={(event) => setCultureInput(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">{isZh ? '自由文本解释（Free-text Rationale）' : 'Free-text rationale'}</span>
              <textarea
                rows={3}
                value={rationale}
                onChange={(event) => setRationale(event.target.value)}
                className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain"
                placeholder={
                  isZh
                    ? '解释文化语境、演奏实践或情感语义（Explain context/performance/affect）...'
                    : 'Explain cultural context, performance practice, or affect semantics...'
                }
              />
            </label>

            <div className="grid gap-2 sm:grid-cols-[1fr_auto] sm:items-center">
              <div className="rounded-xl border border-ink/15 bg-white px-3 py-2 text-xs text-textSub">
                <span className="font-semibold text-textMain">{isZh ? '任务分（Mission Score）' : 'Mission Score'}：</span>
                {(missionScore * 100).toFixed(1)}% · {isZh ? `主动学习增益 ${(activeLearningGain * 100).toFixed(1)}%` : `Gain ${(activeLearningGain * 100).toFixed(1)}%`}
              </div>
              <button onClick={submitAnnotation} className="inline-flex items-center rounded-full bg-zs px-4 py-2 text-sm font-semibold text-white">
                <Send size={14} className="mr-2" />
                {isZh ? '提交标注（Submit）' : 'Submit'}
              </button>
            </div>
          </div>

          <div className="mt-4 space-y-2 text-xs text-textSub">
            {annotations.length === 0 ? <p>{isZh ? '尚未提交标注。' : 'No annotation submitted yet.'}</p> : null}
            {annotations.map((item, index) => (
              <div key={`${item.sampleId}-${index}`} className="rounded-lg border border-ink/15 bg-white px-2 py-1.5">
                {item.sampleId} · {isZh ? cultureLabel(locale, item.cultureLabel) : item.cultureLabel} · {renderAffectLabel(item.affectLabel)} · Q {item.quality.toFixed(2)}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl paper-card p-5">
          <span className="chapter-chip">{isZh ? '本体扩展（Ontology Expansion）' : 'ontology expansion'}</span>
          <h4 className="mt-2 font-display text-xl text-textMain">{isZh ? '动态概念构建器（Dynamic Concept Builder）' : 'Dynamic Concept Builder'}</h4>

          <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_150px_auto]">
            <input
              value={newConcept}
              onChange={(event) => setNewConcept(event.target.value)}
              placeholder={isZh ? '新增概念节点（例如 Han, Saudade）' : 'Add concept node (e.g. Han, Saudade)'}
              className="rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain"
            />
            <select value={anchorCulture} onChange={(event) => setAnchorCulture(event.target.value)} className="rounded-xl border border-ink/20 bg-white px-2 py-2 text-sm text-textMain">
              {cultureNodes.map((node) => (
                <option key={node.id} value={node.name}>
                  {cultureLabel(locale, node.name)}
                </option>
              ))}
            </select>
            <button onClick={addConcept} className="inline-flex items-center justify-center rounded-xl border border-zc/50 bg-zc/10 px-3 py-2 text-zc" aria-label={isZh ? '添加概念节点' : 'Add concept node'}>
              <Plus size={16} />
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {concepts.map((concept) => (
              <span key={concept} className="rounded-full border border-ink/20 bg-white px-2 py-1 text-xs text-textSub">
                {renderConceptLabel(concept)}
                <span className="ml-1 text-[10px] text-textSub/80">→ {cultureLabel(locale, conceptAnchors[concept] ?? 'Western Pop')}</span>
              </span>
            ))}
          </div>

          <div className="mt-4 rounded-xl border border-ink/15 bg-white p-3">
            <div className="mb-2 flex items-center justify-between text-xs text-textSub">
              <span>{isZh ? '反馈趋势（Feedback Trend）' : 'Feedback Trend'}</span>
              <span>{isZh ? `平均覆盖 ${(averageCoverage * 100).toFixed(1)}%` : `Avg coverage ${(averageCoverage * 100).toFixed(1)}%`}</span>
            </div>
            <svg viewBox="0 0 280 92" className="h-[92px] w-full rounded-lg border border-ink/10 bg-[#fff8ee]">
              <polyline points={uncertaintyPoints} fill="none" stroke="rgba(234,67,53,0.85)" strokeWidth="2" />
              <polyline points={coveragePoints} fill="none" stroke="rgba(24,128,56,0.85)" strokeWidth="2" strokeDasharray="5 4" />
            </svg>
            <div className="mt-2 flex items-center gap-3 text-[11px] text-textSub">
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-zc" />
                {isZh ? '不确定性（Uncertainty）' : 'Uncertainty'}
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-zs" />
                {isZh ? '覆盖度（Coverage）' : 'Coverage'}
              </span>
              <span className="inline-flex items-center gap-1">
                <GaugeCircle size={12} />
                {isZh ? `最新质量 Q ${(roundMetrics[roundMetrics.length - 1]?.quality ?? 0).toFixed(2)}` : `Latest quality Q ${(roundMetrics[roundMetrics.length - 1]?.quality ?? 0).toFixed(2)}`}
              </span>
            </div>
          </div>

          <div className="mt-4">
            <p className="mb-2 text-xs text-textSub">{isZh ? '认知正义指标（Cognitive Justice Indicator, Ontology Coverage）' : 'Cognitive Justice Indicator (ontology coverage)'}</p>
            <div className="space-y-2">
              {coverage.map((item) => (
                <div key={item.id}>
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{item.label}</span>
                    <span>{(item.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-gradient-to-r from-zs to-za" style={{ width: `${item.value * 100}%` }} />
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
