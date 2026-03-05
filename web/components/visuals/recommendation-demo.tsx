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
        <div className="paper-card rounded-3xl p-5">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <span className="chapter-chip">ab test</span>
              <h3 className="mt-2 font-display text-3xl text-textMain">Recommendation Lineup</h3>
            </div>
            <span className="sticker">controlled trial</span>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="note-card">
              <div className="mb-2 font-display text-base text-textMain">Conventional Recommender</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {baselineRecommendations.map((item) => (
                  <li key={`b-${item.rank}`} className="flex items-center justify-between rounded-lg border border-ink/10 bg-white px-2 py-1">
                    <span>{item.rank}. {item.title}</span>
                    <span>{item.culture}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-2xl border border-zs/35 bg-zs/10 p-3">
              <div className="mb-2 font-display text-base text-textMain">Soundscape Without Borders</div>
              <ul className="space-y-1.5 text-xs text-textSub">
                {dcasRecommendations.map((item) => (
                  <li key={`d-${item.rank}`} className="flex items-center justify-between rounded-lg border border-zs/20 bg-white px-2 py-1">
                    <span>{item.rank}. {item.title}</span>
                    <span>{item.culture}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className="paper-card rounded-3xl p-5">
          <span className="chapter-chip">serendipity engine</span>
          <h3 className="mt-2 font-display text-3xl text-textMain">Unexpectedness × Relevance</h3>

          <div className="mt-3 space-y-3">
            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>Unexpectedness (zs distance)</span>
                <span>{unexpectedness.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={1} step={0.01} value={unexpectedness} onChange={(event) => setUnexpectedness(Number(event.target.value))} className="w-full accent-zs" />
            </label>

            <label className="block">
              <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                <span>Relevance (za proximity)</span>
                <span>{relevance.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={1} step={0.01} value={relevance} onChange={(event) => setRelevance(Number(event.target.value))} className="w-full accent-za" />
            </label>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="stat-tile">
              <p className="font-mono text-xs text-textSub">Serendipity</p>
              <p className="mt-1 font-display text-4xl text-zs">{serendipity.toFixed(3)}</p>
            </div>
            <div className="stat-tile">
              <p className="font-mono text-xs text-textSub">Decolonization Index</p>
              <p className="mt-1 font-display text-4xl text-za">{(decolonizationIndex * 100).toFixed(1)}%</p>
            </div>
          </div>

          <div className="mt-4 h-2 rounded-full bg-ink/10">
            <div className="h-full rounded-full bg-gradient-to-r from-zc via-zs to-za" style={{ width: `${decolonizationIndex * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="paper-card rounded-3xl p-5">
          <h3 className="font-display text-2xl text-textMain">Cultural Fairness Monitor</h3>
          <div className="mt-3 h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={fairnessData}>
                <CartesianGrid stroke="rgba(70,80,100,.2)" strokeDasharray="3 3" />
                <XAxis dataKey="culture" tick={{ fill: '#5e687d', fontSize: 11 }} />
                <YAxis tick={{ fill: '#5e687d', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(70,80,100,.2)' }} />
                <Legend wrapperStyle={{ color: '#232938' }} />
                <Bar dataKey="baseline" fill="#ff6f61" radius={[5, 5, 0, 0]} />
                <Bar dataKey="dcas" fill="#00a7a0" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="paper-card rounded-3xl p-5">
          <h3 className="font-display text-2xl text-textMain">Culture Distribution</h3>
          <div className="mt-3 h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95} fill="#00a7a0" label />
                <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(70,80,100,.2)' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}