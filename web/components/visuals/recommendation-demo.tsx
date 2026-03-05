'use client'

import { useEffect, useMemo, useState } from 'react'
import { Compass, Dice5, Sparkles } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { cultureNodes, songPoints } from '@/data/mock-data'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { cultureLabel, songTitleLabel } from '@/lib/bilingual'
import { clamp } from '@/lib/utils'

type MoodKey = 'calm' | 'joyful' | 'mystic' | 'melancholic'

type RecommendationItem = {
  id: string
  title: string
  culture: string
  relevance: number
  unexpectedness: number
  fairness: number
  score: number
  reasons: string[]
}

type ScenarioSnapshot = {
  id: string
  serendipity: number
  decolonization: number
  preferredCulture: string
  mood: MoodKey
}

const moodTargets: Record<
  MoodKey,
  {
    en: string
    zh: string
    vector: [number, number]
  }
> = {
  calm: { en: 'Calm', zh: '平静', vector: [-0.55, -0.35] },
  joyful: { en: 'Joyful', zh: '喜悦', vector: [0.68, 0.44] },
  mystic: { en: 'Mystic', zh: '神秘', vector: [0.18, 0.82] },
  melancholic: { en: 'Melancholic', zh: '忧郁', vector: [-0.42, 0.34] }
}

function pairHash(a: string, b: string) {
  const joined = `${a}::${b}`
  let hash = 0
  for (let i = 0; i < joined.length; i += 1) {
    hash = (hash * 31 + joined.charCodeAt(i)) % 997
  }
  return hash / 997
}

function cultureDistance(songCulture: string, preferredCulture: string) {
  if (songCulture === preferredCulture) return 0.08
  return 0.38 + pairHash(songCulture, preferredCulture) * 0.56
}

function normalizeWeights(relevanceWeight: number, noveltyWeight: number, fairnessWeight: number) {
  const total = relevanceWeight + noveltyWeight + fairnessWeight
  if (total <= 0.0001) {
    return {
      relevance: 0.34,
      novelty: 0.33,
      fairness: 0.33
    }
  }

  return {
    relevance: relevanceWeight / total,
    novelty: noveltyWeight / total,
    fairness: fairnessWeight / total
  }
}

function isMainstreamCulture(culture: string) {
  return culture === 'Western Pop'
}

export function RecommendationDemo() {
  const [preferredCulture, setPreferredCulture] = useState<string>('Western Pop')
  const [targetMood, setTargetMood] = useState<MoodKey>('mystic')
  const [exploration, setExploration] = useState(0.6)
  const [relevanceWeight, setRelevanceWeight] = useState(0.45)
  const [noveltyWeight, setNoveltyWeight] = useState(0.3)
  const [fairnessWeight, setFairnessWeight] = useState(0.25)
  const [selectedTrackId, setSelectedTrackId] = useState<string>('')
  const [scenarioLog, setScenarioLog] = useState<ScenarioSnapshot[]>([])

  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const weights = useMemo(() => normalizeWeights(relevanceWeight, noveltyWeight, fairnessWeight), [fairnessWeight, noveltyWeight, relevanceWeight])

  const scored = useMemo(() => {
    const target = moodTargets[targetMood].vector

    return songPoints.map((song) => {
      const dx = song.zaVector[0] - target[0]
      const dy = song.zaVector[1] - target[1]
      const relevance = clamp(1 - Math.sqrt(dx * dx + dy * dy) / 2.2, 0, 1)
      const unexpectedness = clamp(cultureDistance(song.culture, preferredCulture) * (0.46 + exploration * 0.54), 0, 1)
      const fairness = isMainstreamCulture(song.culture) ? 0.15 : 1

      const baselineScore = relevance * 0.78 + (isMainstreamCulture(song.culture) ? 0.18 : 0.03)
      const dcasScore = relevance * weights.relevance + unexpectedness * weights.novelty + fairness * weights.fairness

      const reasons: string[] = []
      if (relevance >= 0.72) {
        reasons.push(isZh ? '情绪邻近（za proximity）' : 'za proximity')
      }
      if (unexpectedness >= 0.62) {
        reasons.push(isZh ? '风格跨度（zs distance）' : 'zs distance')
      }
      if (fairness > 0.8) {
        reasons.push(isZh ? '公平增益（fairness boost）' : 'fairness boost')
      }

      return {
        id: song.id,
        title: song.title,
        culture: song.culture,
        relevance,
        unexpectedness,
        fairness,
        baselineScore,
        dcasScore,
        reasons
      }
    })
  }, [exploration, isZh, preferredCulture, targetMood, weights.fairness, weights.novelty, weights.relevance])

  const baselineList = useMemo<RecommendationItem[]>(() => {
    return [...scored]
      .sort((a, b) => b.baselineScore - a.baselineScore)
      .slice(0, 10)
      .map((item) => ({
        id: item.id,
        title: item.title,
        culture: item.culture,
        relevance: item.relevance,
        unexpectedness: item.unexpectedness,
        fairness: item.fairness,
        score: item.baselineScore,
        reasons: item.reasons
      }))
  }, [scored])

  const dcasList = useMemo<RecommendationItem[]>(() => {
    return [...scored]
      .sort((a, b) => b.dcasScore - a.dcasScore)
      .slice(0, 10)
      .map((item) => ({
        id: item.id,
        title: item.title,
        culture: item.culture,
        relevance: item.relevance,
        unexpectedness: item.unexpectedness,
        fairness: item.fairness,
        score: item.dcasScore,
        reasons: item.reasons
      }))
  }, [scored])

  useEffect(() => {
    if (dcasList.length === 0) return
    setSelectedTrackId((prev) => {
      if (prev && dcasList.some((item) => item.id === prev)) return prev
      return dcasList[0].id
    })
  }, [dcasList])

  const selectedTrack = useMemo(() => dcasList.find((item) => item.id === selectedTrackId) ?? dcasList[0], [dcasList, selectedTrackId])

  const baselineMainstreamShare = useMemo(() => {
    if (baselineList.length === 0) return 0
    return baselineList.filter((item) => isMainstreamCulture(item.culture)).length / baselineList.length
  }, [baselineList])

  const dcasMainstreamShare = useMemo(() => {
    if (dcasList.length === 0) return 0
    return dcasList.filter((item) => isMainstreamCulture(item.culture)).length / dcasList.length
  }, [dcasList])

  const averageUnexpectedness = useMemo(() => {
    if (dcasList.length === 0) return 0
    return dcasList.reduce((acc, item) => acc + item.unexpectedness, 0) / dcasList.length
  }, [dcasList])

  const averageRelevance = useMemo(() => {
    if (dcasList.length === 0) return 0
    return dcasList.reduce((acc, item) => acc + item.relevance, 0) / dcasList.length
  }, [dcasList])

  const serendipity = useMemo(() => Number((averageUnexpectedness * averageRelevance).toFixed(3)), [averageRelevance, averageUnexpectedness])

  const decolonizationIndex = useMemo(() => {
    const minorityShare = 1 - dcasMainstreamShare
    return clamp(minorityShare * 0.62 + serendipity * 0.38, 0, 1)
  }, [dcasMainstreamShare, serendipity])

  const fairnessData = useMemo(
    () => [
      { culture: isZh ? '主流文化（Mainstream）' : 'Mainstream', baseline: baselineMainstreamShare * 100, dcas: dcasMainstreamShare * 100 },
      { culture: isZh ? '少数文化（Minority）' : 'Minority', baseline: (1 - baselineMainstreamShare) * 100, dcas: (1 - dcasMainstreamShare) * 100 }
    ],
    [baselineMainstreamShare, dcasMainstreamShare, isZh]
  )

  const pieData = useMemo(() => {
    const bucket = new Map<string, number>()
    dcasList.forEach((item) => {
      bucket.set(item.culture, (bucket.get(item.culture) ?? 0) + 1)
    })

    return Array.from(bucket.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([culture, count]) => ({
        name: cultureLabel(locale, culture),
        value: count
      }))
  }, [dcasList, locale])

  const captureScenario = () => {
    setScenarioLog((prev) =>
      [
        {
          id: `snap-${Date.now()}`,
          serendipity,
          decolonization: decolonizationIndex,
          preferredCulture,
          mood: targetMood
        },
        ...prev
      ].slice(0, 6)
    )
  }

  const randomizeScenario = () => {
    const randomCulture = cultureNodes[Math.floor(Math.random() * cultureNodes.length)]
    const moods = Object.keys(moodTargets) as MoodKey[]
    const randomMood = moods[Math.floor(Math.random() * moods.length)]

    setPreferredCulture(randomCulture.name)
    setTargetMood(randomMood)
    setExploration(Number((0.3 + Math.random() * 0.7).toFixed(2)))
    setRelevanceWeight(Number((0.25 + Math.random() * 0.5).toFixed(2)))
    setNoveltyWeight(Number((0.2 + Math.random() * 0.55).toFixed(2)))
    setFairnessWeight(Number((0.2 + Math.random() * 0.55).toFixed(2)))
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="paper-card rounded-3xl p-5">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <span className="chapter-chip">{isZh ? 'A/B 对照（AB Control）' : 'ab control'}</span>
              <h3 className="mt-2 font-display text-3xl text-textMain">{isZh ? '推荐列表对比（Recommendation Lineup）' : 'Recommendation Lineup'}</h3>
            </div>
            <span className="sticker">{isZh ? '可执行策略（Executable Strategy）' : 'executable strategy'}</span>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="note-card">
              <div className="mb-2 font-display text-base text-textMain">{isZh ? '传统推荐器（Baseline）' : 'Baseline Recommender'}</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {baselineList.map((item, index) => (
                  <li key={`b-${item.id}`} className="rounded-lg border border-ink/10 bg-white px-2 py-1.5">
                    <div className="flex items-center justify-between">
                      <span>
                        {index + 1}. {songTitleLabel(locale, item.title)}
                      </span>
                      <span>{cultureLabel(locale, item.culture)}</span>
                    </div>
                    <div className="mt-1 text-[10px] text-textSub/80">score {item.score.toFixed(3)}</div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-2xl border border-zs/35 bg-zs/10 p-3">
              <div className="mb-2 font-display text-base text-textMain">{isZh ? '声界无疆（DDRL）' : 'Soundscape Without Borders'}</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {dcasList.map((item, index) => (
                  <li
                    key={`d-${item.id}`}
                    onClick={() => setSelectedTrackId(item.id)}
                    className={`cursor-pointer rounded-lg border px-2 py-1.5 transition ${
                      selectedTrack?.id === item.id ? 'border-zs/45 bg-white' : 'border-zs/20 bg-white/80 hover:bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>
                        {index + 1}. {songTitleLabel(locale, item.title)}
                      </span>
                      <span>{cultureLabel(locale, item.culture)}</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[10px] text-textSub/80">
                      <span>score {item.score.toFixed(3)}</span>
                      <span>{item.reasons.length ? item.reasons[0] : isZh ? '多目标平衡（multi-objective）' : 'multi-objective'}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className="paper-card rounded-3xl p-5">
          <span className="chapter-chip">{isZh ? '策略混音台（Policy Mixer）' : 'policy mixer'}</span>
          <h3 className="mt-2 font-display text-3xl text-textMain">{isZh ? '目标文化 × 情绪 × 权重（Culture × Mood × Weights）' : 'Culture × Mood × Weights'}</h3>

          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-xs text-textSub">{isZh ? '目标文化（Target Culture）' : 'Target Culture'}</span>
              <select value={preferredCulture} onChange={(event) => setPreferredCulture(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain">
                {cultureNodes.map((node) => (
                  <option key={node.id} value={node.name}>
                    {cultureLabel(locale, node.name)}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">{isZh ? '目标情绪（Target Affect）' : 'Target Affect'}</span>
              <select value={targetMood} onChange={(event) => setTargetMood(event.target.value as MoodKey)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain">
                {(Object.keys(moodTargets) as MoodKey[]).map((mood) => (
                  <option key={mood} value={mood}>
                    {isZh ? `${moodTargets[mood].zh}（${moodTargets[mood].en}）` : moodTargets[mood].en}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="mt-3 space-y-3">
            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>{isZh ? '探索强度（Exploration Intensity）' : 'Exploration Intensity'}</span>
                <span>{exploration.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={1} step={0.01} value={exploration} onChange={(event) => setExploration(Number(event.target.value))} className="w-full accent-zc" />
            </label>

            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>{isZh ? '相关性权重（Relevance Weight）' : 'Relevance Weight'}</span>
                <span>{weights.relevance.toFixed(2)}</span>
              </div>
              <input type="range" min={0.05} max={1} step={0.01} value={relevanceWeight} onChange={(event) => setRelevanceWeight(Number(event.target.value))} className="w-full accent-za" />
            </label>

            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>{isZh ? '意外性权重（Unexpectedness Weight）' : 'Unexpectedness Weight'}</span>
                <span>{weights.novelty.toFixed(2)}</span>
              </div>
              <input type="range" min={0.05} max={1} step={0.01} value={noveltyWeight} onChange={(event) => setNoveltyWeight(Number(event.target.value))} className="w-full accent-zs" />
            </label>

            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>{isZh ? '公平性权重（Fairness Weight）' : 'Fairness Weight'}</span>
                <span>{weights.fairness.toFixed(2)}</span>
              </div>
              <input type="range" min={0.05} max={1} step={0.01} value={fairnessWeight} onChange={(event) => setFairnessWeight(Number(event.target.value))} className="w-full accent-zc" />
            </label>
          </div>

          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            <button onClick={randomizeScenario} className="inline-flex items-center justify-center rounded-full border border-ink/20 bg-white px-3 py-2 text-sm font-semibold text-textSub hover:text-textMain">
              <Dice5 size={14} className="mr-2" />
              {isZh ? '随机探索（Randomize）' : 'Randomize'}
            </button>
            <button onClick={captureScenario} className="inline-flex items-center justify-center rounded-full bg-zs px-3 py-2 text-sm font-semibold text-white">
              <Sparkles size={14} className="mr-2" />
              {isZh ? '记录方案（Capture Scenario）' : 'Capture Scenario'}
            </button>
          </div>

          {selectedTrack ? (
            <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-3">
              <p className="font-display text-lg text-textMain">{songTitleLabel(locale, selectedTrack.title)}</p>
              <p className="text-xs text-textSub">{cultureLabel(locale, selectedTrack.culture)}</p>
              <div className="mt-2 space-y-2">
                <div>
                  <div className="mb-1 flex justify-between text-[11px] text-textSub">
                    <span>{isZh ? '相关性（Relevance）' : 'Relevance'}</span>
                    <span>{selectedTrack.relevance.toFixed(2)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-za" style={{ width: `${selectedTrack.relevance * 100}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex justify-between text-[11px] text-textSub">
                    <span>{isZh ? '意外性（Unexpectedness）' : 'Unexpectedness'}</span>
                    <span>{selectedTrack.unexpectedness.toFixed(2)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-zs" style={{ width: `${selectedTrack.unexpectedness * 100}%` }} />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex justify-between text-[11px] text-textSub">
                    <span>{isZh ? '公平增益（Fairness）' : 'Fairness'}</span>
                    <span>{selectedTrack.fairness.toFixed(2)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className="h-full rounded-full bg-zc" style={{ width: `${selectedTrack.fairness * 100}%` }} />
                  </div>
                </div>
              </div>
              <p className="mt-2 text-xs text-textSub">
                <Compass size={12} className="mr-1 inline" />
                {selectedTrack.reasons.length ? selectedTrack.reasons.join(' · ') : isZh ? '多目标权衡（multi-objective balance）' : 'multi-objective balance'}
              </p>
            </div>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="paper-card rounded-3xl p-5">
          <h3 className="font-display text-2xl text-textMain">{isZh ? '文化公平性监测（Cultural Fairness Monitor）' : 'Cultural Fairness Monitor'}</h3>
          <div className="mt-3 h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={fairnessData}>
                <CartesianGrid stroke="rgba(70,80,100,.2)" strokeDasharray="3 3" />
                <XAxis dataKey="culture" tick={{ fill: '#5e687d', fontSize: 11 }} />
                <YAxis tick={{ fill: '#5e687d', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(70,80,100,.2)' }} />
                <Legend wrapperStyle={{ color: '#232938' }} />
                <Bar dataKey="baseline" name={isZh ? '基线（Baseline）' : 'Baseline'} fill="#ea4335" radius={[5, 5, 0, 0]} />
                <Bar dataKey="dcas" name={isZh ? '本系统（DDRL）' : 'DDRL'} fill="#188038" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="paper-card rounded-3xl p-5">
          <h3 className="font-display text-2xl text-textMain">{isZh ? '文化分布与机缘巧合（Distribution & Serendipity）' : 'Distribution & Serendipity'}</h3>
          <div className="mt-3 h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} fill="#188038" label />
                <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(70,80,100,.2)' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            <div className="stat-tile">
              <p className="font-mono text-xs text-textSub">{isZh ? '机缘巧合（Serendipity）' : 'Serendipity'}</p>
              <p className="mt-1 font-display text-4xl text-zs">{serendipity.toFixed(3)}</p>
              <p className="text-[11px] text-textSub">
                {isZh
                  ? `Unexpectedness ${averageUnexpectedness.toFixed(2)} × Relevance ${averageRelevance.toFixed(2)}`
                  : `Unexpectedness ${averageUnexpectedness.toFixed(2)} × Relevance ${averageRelevance.toFixed(2)}`}
              </p>
            </div>
            <div className="stat-tile">
              <p className="font-mono text-xs text-textSub">{isZh ? '去殖民化指数（Decolonization Index）' : 'Decolonization Index'}</p>
              <p className="mt-1 font-display text-4xl text-za">{(decolonizationIndex * 100).toFixed(1)}%</p>
              <p className="text-[11px] text-textSub">{isZh ? `少数文化占比 ${(100 - dcasMainstreamShare * 100).toFixed(1)}%` : `Minority share ${(100 - dcasMainstreamShare * 100).toFixed(1)}%`}</p>
            </div>
          </div>

          <div className="mt-3 h-2 rounded-full bg-ink/10">
            <div className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za" style={{ width: `${decolonizationIndex * 100}%` }} />
          </div>

          <div className="mt-3 space-y-1 text-xs text-textSub">
            {scenarioLog.length === 0 ? (
              <p>{isZh ? '尚未记录方案。' : 'No scenario captured yet.'}</p>
            ) : (
              scenarioLog.map((snapshot) => (
                <p key={snapshot.id}>
                  {cultureLabel(locale, snapshot.preferredCulture)} · {isZh ? `${moodTargets[snapshot.mood].zh}（${moodTargets[snapshot.mood].en}）` : moodTargets[snapshot.mood].en} · S {snapshot.serendipity.toFixed(3)} · D {(snapshot.decolonization * 100).toFixed(1)}%
                </p>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
