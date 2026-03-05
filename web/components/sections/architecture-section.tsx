'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp, cn } from '@/lib/utils'

type StepId = 'A' | 'B' | 'C' | 'D' | 'E'

const impactByStep: Record<
  StepId,
  {
    disentanglement: number
    fairness: number
    fidelity: number
    serendipity: number
  }
> = {
  A: { disentanglement: 0.12, fairness: 0.06, fidelity: 0.11, serendipity: 0.08 },
  B: { disentanglement: 0.24, fairness: 0.08, fidelity: 0.06, serendipity: 0.09 },
  C: { disentanglement: 0.2, fairness: 0.12, fidelity: 0.04, serendipity: 0.07 },
  D: { disentanglement: 0.06, fairness: 0.26, fidelity: 0.02, serendipity: 0.18 },
  E: { disentanglement: 0.05, fairness: 0.18, fidelity: 0.01, serendipity: 0.11 }
}

export function ArchitectureSection({ title }: { title: string }) {
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'
  const [activeStep, setActiveStep] = useState(0)
  const [autoRun, setAutoRun] = useState(false)
  const [componentsEnabled, setComponentsEnabled] = useState<Record<StepId, boolean>>({
    A: true,
    B: true,
    C: true,
    D: true,
    E: true
  })

  const steps: Array<{ id: StepId; title: string; detail: string; accent: string }> = isZh
    ? [
        {
          id: 'A',
          title: 'CultureMERT 主干（CultureMERT Backbone）',
          detail: '在因子分离前先抽取跨流派、跨音色的高质量音频表征，提供稳定先验。',
          accent: 'border-l-zc'
        },
        {
          id: 'B',
          title: '三因子编码器（Tri-Factor Encoder）',
          detail: '将潜变量显式拆分为 zc（内容）、zs（文化/风格）、za（情感）。',
          accent: 'border-l-zs'
        },
        {
          id: 'C',
          title: '泄漏抑制（Leakage Suppression）',
          detail: '通过 GRL + TC + HSIC 约束各通道语义，降低因子串扰。',
          accent: 'border-l-za'
        },
        {
          id: 'D',
          title: '最优传输对齐（OT Alignment）',
          detail: '用 Sinkhorn 正则路径在文化之间搬运偏好质量，构建可解释迁移轨道。',
          accent: 'border-l-zc'
        },
        {
          id: 'E',
          title: 'PAL 反馈回路（PAL Feedback）',
          detail: '高不确定样本进入专家标注，持续重塑潜空间几何与本体覆盖。',
          accent: 'border-l-zs'
        }
      ]
    : [
        {
          id: 'A',
          title: 'CultureMERT Backbone',
          detail: 'Extract rich audio representation with broad genre and timbre priors before factor separation.',
          accent: 'border-l-zc'
        },
        {
          id: 'B',
          title: 'Tri-Factor Encoder',
          detail: 'Split latent channels into zc (content), zs (culture/style), za (affect).',
          accent: 'border-l-zs'
        },
        {
          id: 'C',
          title: 'Leakage Suppression',
          detail: 'GRL + TC + HSIC enforce channel-specific semantics and reduce entanglement.',
          accent: 'border-l-za'
        },
        {
          id: 'D',
          title: 'OT Alignment',
          detail: 'Preference mass is transported across cultures through Sinkhorn-regularized paths.',
          accent: 'border-l-zc'
        },
        {
          id: 'E',
          title: 'PAL Feedback',
          detail: 'Uncertain samples receive expert annotations that reshape latent geometry and ontology coverage.',
          accent: 'border-l-zs'
        }
      ]

  useEffect(() => {
    if (!autoRun) return
    const timer = window.setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length)
    }, 2000)
    return () => window.clearInterval(timer)
  }, [autoRun, steps.length])

  const factorNotes = isZh
    ? [
        { key: 'zc', desc: '旋律 / 节奏骨架（Melody / Rhythm Skeleton）', color: '#ea4335' },
        { key: 'zs', desc: '文化语法与配器（Cultural Grammar + Instrumentation）', color: '#188038' },
        { key: 'za', desc: '效价-唤醒轨迹（Valence-Arousal Trajectory）', color: '#1a73e8' }
      ]
    : [
        { key: 'zc', desc: 'Melody / rhythm skeleton', color: '#ea4335' },
        { key: 'zs', desc: 'Cultural grammar + instrumentation', color: '#188038' },
        { key: 'za', desc: 'Valence-arousal trajectory', color: '#1a73e8' }
      ]

  const enabledCount = useMemo(() => Object.values(componentsEnabled).filter(Boolean).length, [componentsEnabled])

  const dynamicSignals = useMemo(() => {
    const baseline = {
      disentanglement: 0.2,
      fairness: 0.16,
      fidelity: 0.62,
      serendipity: 0.22
    }

    const scored = (Object.keys(componentsEnabled) as StepId[]).reduce(
      (acc, key) => {
        if (!componentsEnabled[key]) return acc
        acc.disentanglement += impactByStep[key].disentanglement
        acc.fairness += impactByStep[key].fairness
        acc.fidelity += impactByStep[key].fidelity
        acc.serendipity += impactByStep[key].serendipity
        return acc
      },
      { ...baseline }
    )

    const stageT = activeStep / Math.max(1, steps.length - 1)

    return {
      disentanglement: clamp(scored.disentanglement + stageT * 0.05, 0, 0.99),
      fairness: clamp(scored.fairness + stageT * 0.04, 0, 0.99),
      fidelity: clamp(scored.fidelity - stageT * 0.05, 0, 0.99),
      serendipity: clamp(scored.serendipity + stageT * 0.06, 0, 0.99)
    }
  }, [activeStep, componentsEnabled, steps.length])

  const liveSignals = useMemo(
    () => [
      {
        label: isZh ? '解耦度（Disentanglement）' : 'Disentanglement',
        value: dynamicSignals.disentanglement,
        color: 'bg-zc'
      },
      {
        label: isZh ? '公平增益（Fairness Gain）' : 'Fairness Gain',
        value: dynamicSignals.fairness,
        color: 'bg-zs'
      },
      {
        label: isZh ? '重建保真（Reconstruction Fidelity）' : 'Reconstruction Fidelity',
        value: dynamicSignals.fidelity,
        color: 'bg-za'
      },
      {
        label: isZh ? '机缘巧合（Serendipity）' : 'Serendipity',
        value: dynamicSignals.serendipity,
        color: 'bg-zc'
      }
    ],
    [dynamicSignals.disentanglement, dynamicSignals.fairness, dynamicSignals.fidelity, dynamicSignals.serendipity, isZh]
  )

  const benchmarkRows = useMemo(
    () => [
      {
        name: isZh ? '标准 VAE（Standard VAE）' : 'Standard VAE',
        disentanglement: 0.31,
        fairness: 0.22,
        serendipity: 0.34
      },
      {
        name: isZh ? 'β-VAE' : 'β-VAE',
        disentanglement: 0.46,
        fairness: 0.28,
        serendipity: 0.41
      },
      {
        name: isZh ? 'FactorVAE' : 'FactorVAE',
        disentanglement: 0.52,
        fairness: 0.33,
        serendipity: 0.45
      },
      {
        name: isZh ? '当前配置（Current Config）' : 'Current Config',
        disentanglement: dynamicSignals.disentanglement,
        fairness: dynamicSignals.fairness,
        serendipity: dynamicSignals.serendipity
      }
    ],
    [dynamicSignals.disentanglement, dynamicSignals.fairness, dynamicSignals.serendipity, isZh]
  )

  const toggleComponent = (id: StepId) => {
    setComponentsEnabled((prev) => {
      const next = { ...prev, [id]: !prev[id] }
      if (Object.values(next).some(Boolean)) return next
      return prev
    })
  }

  return (
    <SectionShell
      id="architecture"
      title={title}
      subtitle={
        isZh
          ? '以课程板（Lesson Board）结构展示：每一块都对应一个训练目标、一个监督信号和一个系统后果。'
          : 'Presented as a lesson board: each block has one objective, one signal, and one downstream consequence.'
      }
      className="min-h-[150vh] bg-[linear-gradient(180deg,rgba(255,255,255,0.35),transparent_35%)]"
    >
      <div className="grid gap-8 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="reveal-item xl:sticky xl:top-24 xl:h-fit">
          <div className="paper-card rounded-3xl p-6">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="chapter-chip">{isZh ? '训练目标（Objective Function）' : 'objective function'}</span>
              <span className="sticker">{isZh ? `${enabledCount}/5 模块启用（Modules ON）` : `${enabledCount}/5 modules on`}</span>
            </div>

            <h3 className="mt-3 font-display text-3xl text-textMain">{isZh ? '三因子学习核心（Three-Factor Learning Core）' : 'Three-Factor Learning Core'}</h3>

            <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-4 font-mono text-xs text-textSub">
              Loss = Recon + beta KL + lambda_domain GRL + lambda_tc TC + lambda_hsic HSIC
            </div>

            <div className="mt-4 grid gap-2">
              {factorNotes.map((item) => (
                <div key={item.key} className="note-card">
                  <p className="font-mono text-[11px] uppercase tracking-[0.14em]" style={{ color: item.color }}>
                    {item.key}
                  </p>
                  <p className="mt-1 text-sm text-textSub">{item.desc}</p>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <p className="font-mono text-xs text-textSub">{isZh ? '流程模拟（Pipeline Simulator）' : 'Pipeline Simulator'}</p>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))}
                    className="rounded-full border border-ink/20 px-2.5 py-1 text-xs text-textSub hover:text-textMain"
                  >
                    {isZh ? '上一步' : 'Prev'}
                  </button>
                  <button
                    onClick={() => setActiveStep((prev) => Math.min(steps.length - 1, prev + 1))}
                    className="rounded-full border border-ink/20 px-2.5 py-1 text-xs text-textSub hover:text-textMain"
                  >
                    {isZh ? '下一步' : 'Next'}
                  </button>
                  <button
                    onClick={() => setAutoRun((prev) => !prev)}
                    className={cn('rounded-full border px-2.5 py-1 text-xs', autoRun ? 'border-zs/40 bg-zs/10 text-zs' : 'border-ink/20 text-textSub hover:text-textMain')}
                  >
                    {isZh ? (autoRun ? '停止自动' : '自动演示') : autoRun ? 'Stop Auto' : 'Auto Play'}
                  </button>
                </div>
              </div>

              <div className="grid gap-2 sm:grid-cols-2">
                {steps.map((step) => (
                  <button
                    key={`switch-${step.id}`}
                    onClick={() => toggleComponent(step.id)}
                    className={cn(
                      'rounded-xl border px-2.5 py-2 text-left text-xs transition',
                      componentsEnabled[step.id] ? 'border-zs/35 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
                    )}
                  >
                    <span className="font-semibold">{step.id}</span> · {componentsEnabled[step.id] ? (isZh ? '启用中（ON）' : 'ON') : isZh ? '已关闭（OFF）' : 'OFF'}
                  </button>
                ))}
              </div>

              <div className="mt-3 space-y-2">
                {liveSignals.map((signal) => (
                  <div key={signal.label}>
                    <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                      <span>{signal.label}</span>
                      <span>{(signal.value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-ink/10">
                      <div className={cn('h-full rounded-full', signal.color)} style={{ width: `${signal.value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 rounded-2xl border border-zs/30 bg-zs/10 p-3 text-sm text-textMain">
              {isZh
                ? '评估目标：在保持跨文化语义一致性的前提下，同时提升机缘巧合性（Serendipity）与少数文化曝光。'
                : 'Evaluation target: raise serendipity and minority exposure while preserving cross-cultural semantic fidelity.'}
            </div>

            <div className="mt-4 rounded-2xl border border-ink/15 bg-white p-3">
              <p className="font-mono text-xs text-textSub">{isZh ? '对比基线（Baseline Comparison）' : 'Baseline Comparison'}</p>
              <div className="mt-2 space-y-2 text-xs text-textSub">
                {benchmarkRows.map((row) => (
                  <div key={row.name} className="rounded-xl border border-ink/12 bg-white px-2.5 py-2">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-textMain">{row.name}</span>
                      <span>S {(row.serendipity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-ink/10">
                      <div className="h-full rounded-full bg-zc" style={{ width: `${row.disentanglement * 100}%` }} />
                    </div>
                    <div className="mt-1 h-1 rounded-full bg-ink/10">
                      <div className="h-full rounded-full bg-zs" style={{ width: `${row.fairness * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {steps.map((step, index) => {
            const enabled = componentsEnabled[step.id]
            return (
              <motion.article
                key={step.id}
                onClick={() => setActiveStep(index)}
                className={cn(
                  `reveal-item rounded-3xl border border-ink/15 bg-white/85 p-5 ${step.accent} border-l-[6px] ${index % 2 ? 'md:ml-10' : 'md:mr-10'} cursor-pointer`,
                  activeStep === index ? 'ring-2 ring-za/30' : '',
                  enabled ? '' : 'opacity-45 grayscale'
                )}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-10% 0px -10% 0px' }}
                transition={{ duration: 0.45, delay: index * 0.06 }}
              >
                <div className="flex items-start gap-3">
                  <span className={cn('inline-flex h-8 w-8 items-center justify-center rounded-xl font-mono text-xs', activeStep === index ? 'bg-za/15 text-za' : 'bg-ink/5 text-textSub')}>
                    {step.id}
                  </span>
                  <div className="w-full">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h4 className="font-display text-2xl text-textMain">{step.title}</h4>
                      <span className={cn('rounded-full border px-2 py-0.5 text-[11px]', enabled ? 'border-zs/35 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub')}>
                        {enabled ? (isZh ? '启用（ON）' : 'ON') : isZh ? '关闭（OFF）' : 'OFF'}
                      </span>
                    </div>
                    <p className="mt-2 text-base leading-relaxed text-textSub">{step.detail}</p>
                    <div className="mt-3 h-1.5 rounded-full bg-ink/10">
                      <div className="h-full rounded-full bg-za transition-all" style={{ width: `${activeStep >= index && enabled ? 100 : enabled ? 28 : 8}%` }} />
                    </div>
                  </div>
                </div>
              </motion.article>
            )
          })}
        </div>
      </div>
    </SectionShell>
  )
}
