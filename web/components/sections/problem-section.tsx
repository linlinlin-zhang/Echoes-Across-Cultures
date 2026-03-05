'use client'

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { SectionShell } from '@/components/layout/section-shell'

const babelData = [
  { phase: 'In-domain', relevance: 0.86, culturalFairness: 0.42 },
  { phase: 'Cross-domain', relevance: 0.61, culturalFairness: 0.27 },
  { phase: 'Long-tail', relevance: 0.48, culturalFairness: 0.18 },
  { phase: 'DDRL+OT', relevance: 0.79, culturalFairness: 0.71 }
]

export function ProblemSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="problem"
      title={title}
      subtitle="Conventional recommenders collapse diverse ontologies into a dominant latent axis. DDRL separates content, culture, and affect to restore representational justice."
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <div className="space-y-4">
          {[
            {
              heading: 'Representation Collapse',
              body: 'Multiple cultural grammars are projected into one homogenized latent manifold.'
            },
            {
              heading: 'Affective Misalignment',
              body: 'Emotion transfer often ignores culturally specific affect semantics and ritual context.'
            },
            {
              heading: 'Exposure Inequity',
              body: 'Mainstream catalogs dominate recommendation slots, reducing minority cultural visibility.'
            }
          ].map((item) => (
            <article key={item.heading} className="reveal-item rounded-2xl border border-white/10 bg-black/25 p-5">
              <h3 className="font-display text-xl text-textMain">{item.heading}</h3>
              <p className="mt-2 font-body text-base leading-relaxed text-textSub">{item.body}</p>
            </article>
          ))}
        </div>

        <div className="reveal-item rounded-3xl border border-white/10 bg-black/30 p-4">
          <div className="mb-3 font-display text-xl text-textMain">Digital Babel Stress Test</div>
          <div className="h-[360px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={babelData} margin={{ left: 4, right: 8, top: 12, bottom: 12 }}>
                <defs>
                  <linearGradient id="relevance" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ff6b6b" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#ff6b6b" stopOpacity={0.1} />
                  </linearGradient>
                  <linearGradient id="fairness" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4ecdc4" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#4ecdc4" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                <XAxis dataKey="phase" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: '#0f172a',
                    border: '1px solid rgba(148,163,184,.4)',
                    color: '#e2e8f0'
                  }}
                />
                <Area type="monotone" dataKey="relevance" stroke="#ff6b6b" fill="url(#relevance)" strokeWidth={2} />
                <Area
                  type="monotone"
                  dataKey="culturalFairness"
                  stroke="#4ecdc4"
                  fill="url(#fairness)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </SectionShell>
  )
}
