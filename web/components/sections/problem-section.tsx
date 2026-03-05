'use client'

import { useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { SectionShell } from '@/components/layout/section-shell'
import { cn } from '@/lib/utils'

const babelData = [
  { phase: 'In-domain', relevance: 0.86, culturalFairness: 0.42, uncertainty: 0.18 },
  { phase: 'Cross-domain', relevance: 0.61, culturalFairness: 0.27, uncertainty: 0.44 },
  { phase: 'Long-tail', relevance: 0.48, culturalFairness: 0.18, uncertainty: 0.62 },
  { phase: 'DDRL+OT', relevance: 0.79, culturalFairness: 0.71, uncertainty: 0.24 }
]

const chapters = [
  {
    title: 'Semantic Collapse',
    body: 'Many recommenders reduce all cultural grammars to a single manifold. Distinct epistemologies are flattened into one “distance”.',
    cue: 'What is easy to rank is not equal to what is culturally meaningful.'
  },
  {
    title: 'Affective Drift',
    body: 'Emotion tags are transferred across cultures without context. Similar valence labels can encode very different social and ritual functions.',
    cue: 'Shared labels do not imply shared meaning.'
  },
  {
    title: 'Exposure Inequality',
    body: 'Minority traditions remain low-exposure tails in top-N recommendations, even when users show exploratory intent.',
    cue: 'Discovery without redistribution is cosmetic diversity.'
  },
  {
    title: 'DDRL Response',
    body: 'Disentangle latent factors first, align with OT second, then optimize recommendations under fairness-aware objectives.',
    cue: 'Interpretability is treated as a system constraint, not a by-product.'
  }
]

export function ProblemSection({ title }: { title: string }) {
  const [active, setActive] = useState(0)

  const highlight = useMemo(() => babelData[Math.min(active, babelData.length - 1)], [active])

  return (
    <SectionShell
      id="problem"
      title={title}
      subtitle="Like data essays on Pudding, this chapter invites readers to inspect failure modes as a narrative, not a static benchmark table."
    >
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="reveal-item lg:sticky lg:top-24 lg:h-fit">
          <div className="paper-card rounded-3xl p-5">
            <p className="chapter-chip">digital babel monitor</p>
            <h3 className="mt-3 font-display text-3xl text-textMain">{highlight.phase}</h3>

            <div className="mt-4 grid grid-cols-3 gap-2">
              {[
                { label: 'Relevance', value: highlight.relevance, color: 'bg-zc' },
                { label: 'Fairness', value: highlight.culturalFairness, color: 'bg-zs' },
                { label: 'Uncertainty', value: highlight.uncertainty, color: 'bg-za' }
              ].map((item) => (
                <div key={item.label} className="stat-tile">
                  <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-textSub">{item.label}</p>
                  <p className="mt-1 font-display text-lg text-textMain">{(item.value * 100).toFixed(0)}%</p>
                  <div className="mt-1 h-1.5 rounded-full bg-ink/10">
                    <div className={cn('h-full rounded-full', item.color)} style={{ width: `${item.value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 h-[270px] rounded-2xl border border-ink/15 bg-white p-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={babelData} margin={{ left: 2, right: 2, top: 8, bottom: 4 }}>
                  <defs>
                    <linearGradient id="rel" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ff6f61" stopOpacity={0.78} />
                      <stop offset="100%" stopColor="#ff6f61" stopOpacity={0.08} />
                    </linearGradient>
                    <linearGradient id="fair" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00a7a0" stopOpacity={0.76} />
                      <stop offset="100%" stopColor="#00a7a0" stopOpacity={0.08} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(70,80,100,.22)" strokeDasharray="3 3" />
                  <XAxis dataKey="phase" tick={{ fill: '#576178', fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#576178', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(60,70,90,.25)' }} />
                  <Area type="monotone" dataKey="relevance" stroke="#ff6f61" fill="url(#rel)" strokeWidth={2} />
                  <Area type="monotone" dataKey="culturalFairness" stroke="#00a7a0" fill="url(#fair)" strokeWidth={2} />
                  <ReferenceDot x={highlight.phase} y={highlight.relevance} fill="#232938" r={4} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {chapters.map((chapter, index) => (
            <article
              key={chapter.title}
              onMouseEnter={() => setActive(index)}
              className={cn(
                'reveal-item cursor-default rounded-3xl border p-6 transition',
                active === index ? 'border-zs/45 bg-zs/10 shadow-[0_8px_20px_rgba(0,167,160,.18)]' : 'paper-card hover:border-ink/35'
              )}
            >
              <span className="chapter-chip">chapter {String(index + 1).padStart(2, '0')}</span>
              <h3 className="mt-3 font-display text-3xl text-textMain">{chapter.title}</h3>
              <p className="mt-3 text-base leading-relaxed text-textSub">{chapter.body}</p>
              <div className="mt-4 rounded-xl border border-ink/15 bg-white px-3 py-2 font-mono text-xs text-textSub">{chapter.cue}</div>
            </article>
          ))}
        </div>
      </div>
    </SectionShell>
  )
}