'use client'

import { motion } from 'framer-motion'

import { SectionShell } from '@/components/layout/section-shell'

const steps = [
  {
    id: 'A',
    title: 'CultureMERT Backbone',
    detail: 'Extract rich audio representation with broad genre and timbre priors before factor separation.'
  },
  {
    id: 'B',
    title: 'Tri-Factor Encoder',
    detail: 'Split latent channels into zc (content), zs (culture/style), za (affect).' 
  },
  {
    id: 'C',
    title: 'Leakage Suppression',
    detail: 'GRL + TC + HSIC enforce channel-specific semantics and reduce entanglement.'
  },
  {
    id: 'D',
    title: 'OT Alignment',
    detail: 'Preference mass is transported across cultures through Sinkhorn-regularized paths.'
  },
  {
    id: 'E',
    title: 'PAL Feedback',
    detail: 'Uncertain samples receive expert annotations that reshape latent geometry and ontology coverage.'
  }
]

export function ArchitectureSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="architecture"
      title={title}
      subtitle="Presented as a lesson board: each block has one objective, one signal, and one downstream consequence."
      className="min-h-[150vh]"
    >
      <div className="grid gap-8 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="reveal-item xl:sticky xl:top-24 xl:h-fit">
          <div className="paper-card rounded-3xl p-6">
            <span className="chapter-chip">objective function</span>
            <h3 className="mt-3 font-display text-3xl text-textMain">Three-Factor Learning Core</h3>

            <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-4 font-mono text-xs text-textSub">
              Loss = Recon + beta KL + lambda_domain GRL + lambda_tc TC + lambda_hsic HSIC
            </div>

            <div className="mt-4 grid gap-2">
              {[
                { key: 'zc', desc: 'Melody / rhythm skeleton', color: '#ff6f61' },
                { key: 'zs', desc: 'Cultural grammar + instrumentation', color: '#00a7a0' },
                { key: 'za', desc: 'Valence-arousal trajectory', color: '#7e57c2' }
              ].map((item) => (
                <div key={item.key} className="note-card">
                  <p className="font-mono text-[11px] uppercase tracking-[0.14em]" style={{ color: item.color }}>{item.key}</p>
                  <p className="mt-1 text-sm text-textSub">{item.desc}</p>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-2xl border border-zs/30 bg-zs/10 p-3 text-sm text-textMain">
              Evaluation target: raise serendipity and minority exposure while preserving cross-cultural semantic fidelity.
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {steps.map((step, index) => (
            <motion.article
              key={step.id}
              className="reveal-item paper-card rounded-3xl p-5"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-10% 0px -10% 0px' }}
              transition={{ duration: 0.45, delay: index * 0.06 }}
            >
              <div className="flex items-start gap-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl border border-ink/20 bg-white font-mono text-xs text-textSub">{step.id}</span>
                <div>
                  <h4 className="font-display text-2xl text-textMain">{step.title}</h4>
                  <p className="mt-2 text-base leading-relaxed text-textSub">{step.detail}</p>
                </div>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </SectionShell>
  )
}