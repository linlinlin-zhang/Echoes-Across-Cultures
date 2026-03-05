'use client'

import { motion } from 'framer-motion'

import { SectionShell } from '@/components/layout/section-shell'

const steps = [
  {
    id: 'S1',
    title: 'Backbone Embedding',
    detail: 'CultureMERT projects audio into a high-capacity shared representation.'
  },
  {
    id: 'S2',
    title: 'Disentanglement Encoder',
    detail: 'Variational heads split latent variables into zc (content), zs (style), za (affect).'
  },
  {
    id: 'S3',
    title: 'Domain Adversarial Alignment',
    detail: 'GRL discourages culture leakage in affect channel while preserving semantic salience.'
  },
  {
    id: 'S4',
    title: 'OT Recommendation Bridge',
    detail: 'Sinkhorn transport maps user preference manifolds toward target-culture candidate manifolds.'
  },
  {
    id: 'S5',
    title: 'PAL Feedback Loop',
    detail: 'Experts annotate uncertain regions; constraints are fed back to reshape latent geometry.'
  }
]

export function ArchitectureSection({ title }: { title: string }) {
  return (
    <SectionShell
      id="architecture"
      title={title}
      subtitle="Scroll through the computational pipeline from feature extraction to recommendation and participatory feedback."
      className="min-h-[150vh]"
    >
      <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="sticky top-24 h-fit rounded-3xl border border-white/10 bg-black/30 p-6">
          <h3 className="font-display text-2xl text-textMain">Three-Factor Latent Objective</h3>
          <div className="mt-4 space-y-3 font-mono text-xs text-textSub">
            <p>Loss = Recon + beta*KL + lambda_domain*GRL + lambda_tc*TC + lambda_hsic*HSIC</p>
            <p className="text-zc">zc → melodic/rhythmic invariants</p>
            <p className="text-zs">zs → culture/style signatures</p>
            <p className="text-za">za → affective trajectory</p>
          </div>
          <div className="mt-5 rounded-xl border border-zs/30 bg-zs/10 p-3 text-sm text-textMain">
            Objective: maximize serendipity while maintaining cultural calibration and minority exposure.
          </div>
        </div>

        <div className="space-y-5">
          {steps.map((step, index) => (
            <motion.article
              key={step.id}
              className="reveal-item rounded-2xl border border-white/10 bg-black/25 p-5"
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-10% 0px -10% 0px' }}
              transition={{ duration: 0.45, delay: index * 0.08 }}
            >
              <div className="mb-2 inline-flex rounded-full bg-white/10 px-2 py-1 font-mono text-xs text-textSub">
                {step.id}
              </div>
              <h4 className="font-display text-xl text-textMain">{step.title}</h4>
              <p className="mt-2 font-body text-base text-textSub">{step.detail}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </SectionShell>
  )
}
