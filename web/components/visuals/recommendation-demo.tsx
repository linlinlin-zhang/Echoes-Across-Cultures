'use client'

import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { baselineRecommendations, dcasRecommendations } from '@/data/mock-data'
import { clamp } from '@/lib/utils'

const fairnessData = [
  { culture: 'Mainstream', baseline: 74, dcas: 41 },
  { culture: 'Minority', baseline: 26, dcas: 59 }
]

const pieData = [
  { name: 'Western Pop', value: 19 },
  { name: 'Indian Classical', value: 14 },
  { name: 'Turkish Makam', value: 12 },
  { name: 'Guqin', value: 9 },
  { name: 'Arabic Maqam', value: 8 },
  { name: 'Other', value: 11 }
]

export function RecommendationDemo() {
  const [unexpectedness, setUnexpectedness] = useState(0.64)
  const [relevance, setRelevance] = useState(0.72)

  const serendipity = useMemo(() => Number((unexpectedness * relevance).toFixed(3)), [unexpectedness, relevance])
  const decolonizationIndex = useMemo(() => clamp((59 / 41) * 0.42 + serendipity * 0.6, 0, 1), [serendipity])

  return (
    <div className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-3xl panel-deep p-5">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-textSub">A/B Deck</p>
              <h3 className="font-display text-2xl text-textMain">Recommendation Lineup</h3>
            </div>
            <span className="rounded-full border border-zs/40 bg-zs/10 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em] text-zs">
              Controlled Trial
            </span>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-black/25 p-3">
              <div className="mb-2 font-display text-sm text-textMain">Conventional Recommender</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {baselineRecommendations.map((item) => (
                  <li key={`b-${item.rank}`} className="flex items-center justify-between rounded-lg bg-white/[0.03] px-2 py-1">
                    <span>
                      {item.rank}. {item.title}
                    </span>
                    <span>{item.culture}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-2xl border border-zs/35 bg-zs/10 p-3">
              <div className="mb-2 font-display text-sm text-textMain">Soundscape Without Borders</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {dcasRecommendations.map((item) => (
                  <li key={`d-${item.rank}`} className="flex items-center justify-between rounded-lg bg-black/25 px-2 py-1">
                    <span>
                      {item.rank}. {item.title}
                    </span>
                    <span>{item.culture}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className="rounded-3xl panel-deep p-5">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-textSub">Serendipity Engine</p>
          <h3 className="font-display text-2xl text-textMain">Unexpectedness × Relevance</h3>

          <div className="mt-3 space-y-3">
            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>Unexpectedness (zs distance)</span>
                <span>{unexpectedness.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={unexpectedness}
                onChange={(event) => setUnexpectedness(Number(event.target.value))}
                className="w-full accent-zs"
              />
            </label>

            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>Relevance (za proximity)</span>
                <span>{relevance.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={relevance}
                onChange={(event) => setRelevance(Number(event.target.value))}
                className="w-full accent-za"
              />
            </label>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-black/25 p-4">
              <p className="font-mono text-xs text-textSub">Serendipity Score</p>
              <p className="mt-1 font-display text-4xl text-zs">{serendipity.toFixed(3)}</p>
            </div>
            <div className="rounded-xl border border-white/10 bg-black/25 p-4">
              <p className="font-mono text-xs text-textSub">Decolonization Index</p>
              <p className="mt-1 font-display text-4xl text-za">{(decolonizationIndex * 100).toFixed(1)}%</p>
            </div>
          </div>

          <div className="mt-4 h-2 rounded-full bg-white/10">
            <div className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za" style={{ width: `${decolonizationIndex * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-3xl panel-deep p-5">
          <h3 className="font-display text-xl text-textMain">Cultural Fairness Monitor</h3>
          <div className="mt-3 h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={fairnessData}>
                <CartesianGrid stroke="rgba(148,163,184,.25)" strokeDasharray="3 3" />
                <XAxis dataKey="culture" tick={{ fill: '#d7e0ff', fontSize: 11 }} />
                <YAxis tick={{ fill: '#d7e0ff', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f1427', border: '1px solid rgba(148,163,184,.4)' }} />
                <Legend wrapperStyle={{ color: '#e2e8f0' }} />
                <Bar dataKey="baseline" fill="#ff6b6b" radius={[5, 5, 0, 0]} />
                <Bar dataKey="dcas" fill="#4ecdc4" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-3xl panel-deep p-5">
          <h3 className="font-display text-xl text-textMain">Recommendation Culture Distribution</h3>
          <div className="mt-3 h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={95}
                  fill="#4ecdc4"
                  label
                />
                <Tooltip contentStyle={{ background: '#0f1427', border: '1px solid rgba(148,163,184,.4)' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}