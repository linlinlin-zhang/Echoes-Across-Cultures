'use client'

import { useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

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
    title: 'Collapse of Ontologies',
    body: 'Mainstream pipelines compress many musical epistemologies into one axis, then call it similarity.',
    cue: 'Representation variance shrinks as culture diversity rises.'
  },
  {
    title: 'Affective Misread',
    body: 'Emotion labels are transferred without cultural grounding, producing false equivalence across contexts.',
    cue: 'Valence is treated as universal even when semantics diverge.'
  },
  {
    title: 'Exposure Asymmetry',
    body: 'Minority cultures become decorative outliers in top-N lists instead of first-class recommendation targets.',
    cue: 'Recommendation mass remains trapped in dominant catalogs.'
  },
  {
    title: 'Why DDRL + OT',
    body: 'Disentangle first, align second, then optimize recommendation flow under fairness constraints.',
    cue: 'Serendipity increases without sacrificing cultural integrity.'
  }
]

export function ProblemSection({ title }: { title: string }) {
  const [active, setActive] = useState(0)

  const highlight = useMemo(() => babelData[Math.min(active, babelData.length - 1)], [active])

  return (
    <SectionShell
      id="problem"
      title={title}
      subtitle="Inspired by narrative data essays, this section turns model failure into an explorable story instead of a static paragraph."
    >
      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="reveal-item lg:sticky lg:top-24 lg:h-fit">
          <div className="rounded-3xl panel-deep p-5">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-textSub">Digital Babel Monitor</p>
            <h3 className="mt-2 font-display text-2xl text-textMain">{highlight.phase}</h3>

            <div className="mt-4 grid grid-cols-3 gap-2">
              {[
                { label: 'Relevance', value: highlight.relevance, color: 'bg-zc' },
                { label: 'Fairness', value: highlight.culturalFairness, color: 'bg-zs' },
                { label: 'Uncertainty', value: highlight.uncertainty, color: 'bg-za' }
              ].map((item) => (
                <div key={item.label} className="rounded-xl border border-white/10 bg-black/25 p-2">
                  <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-textSub">{item.label}</p>
                  <p className="mt-1 font-display text-lg text-textMain">{(item.value * 100).toFixed(0)}%</p>
                  <div className="mt-1 h-1.5 rounded-full bg-white/10">
                    <div className={cn('h-full rounded-full', item.color)} style={{ width: `${item.value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 h-[260px] w-full rounded-2xl border border-white/10 bg-black/20 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={babelData} margin={{ left: 2, right: 2, top: 8, bottom: 4 }}>
                  <defs>
                    <linearGradient id="rel" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ff6b6b" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="#ff6b6b" stopOpacity={0.08} />
                    </linearGradient>
                    <linearGradient id="fair" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4ecdc4" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="#4ecdc4" stopOpacity={0.08} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(148,163,184,.28)" strokeDasharray="3 3" />
                  <XAxis dataKey="phase" tick={{ fill: '#c8d2f0', fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#c8d2f0', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#0f1427', border: '1px solid rgba(148,163,184,.5)' }} />
                  <Area type="monotone" dataKey="relevance" stroke="#ff6b6b" fill="url(#rel)" strokeWidth={2} />
                  <Area type="monotone" dataKey="culturalFairness" stroke="#4ecdc4" fill="url(#fair)" strokeWidth={2} />
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
                active === index
                  ? 'border-zs/55 bg-zs/10 shadow-[0_0_24px_rgba(78,205,196,0.15)]'
                  : 'border-white/10 bg-black/25 hover:border-white/30'
              )}
            >
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-textSub">Chapter {String(index + 1).padStart(2, '0')}</p>
              <h3 className="mt-2 font-display text-2xl text-textMain">{chapter.title}</h3>
              <p className="mt-3 text-base leading-relaxed text-textSub">{chapter.body}</p>
              <div className="mt-4 rounded-xl border border-white/10 bg-black/25 px-3 py-2 font-mono text-xs text-textSub">{chapter.cue}</div>
            </article>
          ))}
        </div>
      </div>
    </SectionShell>
  )
}