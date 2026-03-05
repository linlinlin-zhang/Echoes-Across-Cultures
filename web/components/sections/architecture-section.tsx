'use client'

import { motion } from 'framer-motion'

import { SectionShell } from '@/components/layout/section-shell'

const steps = [
  {
    id: '01',
    title: 'CultureMERT Backbone',
    detail: 'High-capacity embedding stage that preserves timbre, rhythm, and context priors before disentanglement.'
  },
  {
    id: '02',
    title: 'Tri-Factor VAE Heads',
    detail: 'Three dedicated latent channels isolate content, style/culture, and affect with explicit inductive constraints.'
  },
  {
    id: '03',
    title: 'Adversarial Decorrelation',
    detail: 'GRL + TC + HSIC suppress leakage so each factor stays semantically identifiable.'
  },
  {
    id: '04',
    title: 'Optimal Transport Bridge',
    detail: 'Sinkhorn flow aligns user preference mass from source manifold to target-culture candidates.'
  },
  {
    id: '05',
    title: 'PAL Constraint Return',
    detail: 'Expert feedback is written back as metric constraints and ontology updates, reshaping latent geometry.'
  }
]

export function ArchitectureSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="architecture"
      title={title}
      subtitle="Built like a mission-control sequence: each stage has a clear objective, measurable signal, and downstream effect."
      className="min-h-[150vh]"
    >
      <div className="grid gap-8 lg:grid-cols-[0.88fr_1.12fr]">
        <div className="reveal-item lg:sticky lg:top-24 lg:h-fit">
          <div className="rounded-3xl panel-deep p-6">
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-textSub">Objective Matrix</p>
            <h3 className="mt-2 font-display text-2xl text-textMain">Three-Factor Learning Core</h3>

            <div className="mt-4 rounded-2xl border border-white/10 bg-black/25 p-4 font-mono text-xs text-textSub">
              Loss = Recon + beta KL + lambda_domain GRL + lambda_tc TC + lambda_hsic HSIC
            </div>

            <div className="mt-4 space-y-3">
              {[
                { name: 'zc / content', desc: 'melody-rhythm skeleton', color: '#ff6b6b' },
                { name: 'zs / culture', desc: 'instrumentation + grammar', color: '#4ecdc4' },
                { name: 'za / affect', desc: 'valence-arousal dynamics', color: '#a55eea' }
              ].map((item) => (
                <div key={item.name} className="rounded-xl border border-white/10 bg-black/20 p-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.14em]" style={{ color: item.color }}>
                    {item.name}
                  </p>
                  <p className="mt-1 text-sm text-textSub">{item.desc}</p>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-xl border border-zs/35 bg-zs/10 p-3 text-sm text-textMain">
              Target state: maximize cross-cultural serendipity while preserving epistemic plurality and exposure fairness.
            </div>
          </div>
        </div>

        <div className="relative space-y-4">
          <div className="absolute left-[22px] top-2 hidden h-[calc(100%-16px)] w-px bg-gradient-to-b from-zc via-zs to-za md:block" />
          {steps.map((step, index) => (
            <motion.article
              key={step.id}
              className="reveal-item relative rounded-3xl border border-white/10 bg-black/25 p-5 md:pl-16"
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-10% 0px -10% 0px' }}
              transition={{ duration: 0.45, delay: index * 0.07 }}
            >
              <div className="mb-2 inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/20 bg-black/50 font-mono text-xs text-textSub md:absolute md:left-2 md:top-5">
                {step.id}
              </div>
              <h4 className="font-display text-2xl text-textMain">{step.title}</h4>
              <p className="mt-3 text-base leading-relaxed text-textSub">{step.detail}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </SectionShell>
  )
}