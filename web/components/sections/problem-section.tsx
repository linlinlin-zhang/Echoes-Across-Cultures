'use client'

import { useEffect, useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { clamp, cn } from '@/lib/utils'

type BabelPoint = {
  phase: string
  phaseLabelZh: string
  relevance: number
  culturalFairness: number
  uncertainty: number
}

type PolicyMode = 'baseline' | 'balanced' | 'fairness'

const baseBabelData: BabelPoint[] = [
  { phase: 'In-domain', phaseLabelZh: '领域内（In-domain）', relevance: 0.86, culturalFairness: 0.42, uncertainty: 0.18 },
  { phase: 'Cross-domain', phaseLabelZh: '跨域（Cross-domain）', relevance: 0.61, culturalFairness: 0.27, uncertainty: 0.44 },
  { phase: 'Long-tail', phaseLabelZh: '长尾（Long-tail）', relevance: 0.48, culturalFairness: 0.18, uncertainty: 0.62 },
  { phase: 'DDRL+OT', phaseLabelZh: '深度解纠缠 + 最优传输（DDRL+OT）', relevance: 0.79, culturalFairness: 0.71, uncertainty: 0.24 }
]

const policyAdjustments: Record<PolicyMode, { relevance: number; fairness: number; uncertainty: number }> = {
  baseline: { relevance: 0.06, fairness: -0.12, uncertainty: 0.08 },
  balanced: { relevance: 0.02, fairness: 0.02, uncertainty: -0.01 },
  fairness: { relevance: -0.05, fairness: 0.15, uncertainty: -0.06 }
}

export function ProblemSection({ title }: { title: string }) {
  const [active, setActive] = useState(0)
  const [policyMode, setPolicyMode] = useState<PolicyMode>('balanced')
  const [explorationIntent, setExplorationIntent] = useState(0.58)
  const [storyAutoPlay, setStoryAutoPlay] = useState(false)
  const [interventionEnabled, setInterventionEnabled] = useState(true)
  const [repairBudget, setRepairBudget] = useState(0.66)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  useEffect(() => {
    if (!storyAutoPlay) return
    const timer = window.setInterval(() => {
      setActive((prev) => (prev + 1) % 4)
    }, 2400)
    return () => window.clearInterval(timer)
  }, [storyAutoPlay])

  const chapters = useMemo(
    () =>
      isZh
        ? [
            {
              title: '语义坍缩（Semantic Collapse）',
              body: '许多推荐系统把不同文化语法压缩到单一流形中。不同知识体系被扁平化为同一种“距离”。',
              cue: '易于排序（Easy to Rank）不等于文化上有意义（Culturally Meaningful）。'
            },
            {
              title: '情感漂移（Affective Drift）',
              body: '情感标签在跨文化迁移时常缺少语境。相同 valence 标签在不同文化中可能对应完全不同的社会与仪式功能。',
              cue: '标签共享（Shared Labels）并不代表语义共享（Shared Meaning）。'
            },
            {
              title: '曝光不平等（Exposure Inequality）',
              body: '即使用户有探索意图，少数文化传统在 Top-N 推荐中仍长期处于低曝光尾部。',
              cue: '没有再分配的“发现”（Discovery）只是表层多样性（Cosmetic Diversity）。'
            },
            {
              title: 'DDRL 应对策略（DDRL Response）',
              body: '先做潜因子解耦，再做最优传输对齐，最后在公平约束下优化推荐目标。',
              cue: '可解释性（Interpretability）是系统约束，不是附带产物。'
            }
          ]
        : [
            {
              title: 'Semantic Collapse',
              body: 'Many recommenders reduce all cultural grammars to a single manifold. Distinct epistemologies are flattened into one distance.',
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
          ],
    [isZh]
  )

  const babelData = useMemo(() => {
    const policy = policyAdjustments[policyMode]
    const intentShift = explorationIntent - 0.5
    const budgetShift = repairBudget - 0.5

    return baseBabelData.map((item, index) => {
      const longTailBoost = index >= 2 ? 1.2 : 0.7

      let relevance = clamp(item.relevance + policy.relevance + intentShift * 0.16 * (index === 0 ? -0.5 : 1), 0.05, 0.98)
      let fairness = clamp(item.culturalFairness + policy.fairness + intentShift * 0.2 * longTailBoost, 0.05, 0.98)
      let uncertainty = clamp(item.uncertainty + policy.uncertainty - intentShift * 0.12 * longTailBoost, 0.03, 0.99)

      if (index === 3) {
        if (interventionEnabled) {
          relevance = clamp(relevance + 0.12 + budgetShift * 0.2, 0.05, 0.99)
          fairness = clamp(fairness + 0.2 + budgetShift * 0.28, 0.05, 0.99)
          uncertainty = clamp(uncertainty - 0.18 - budgetShift * 0.2, 0.02, 0.99)
        } else {
          relevance = clamp(baseBabelData[2].relevance + policy.relevance - 0.03 + intentShift * 0.08, 0.05, 0.98)
          fairness = clamp(baseBabelData[2].culturalFairness + policy.fairness - 0.04 + intentShift * 0.05, 0.05, 0.96)
          uncertainty = clamp(baseBabelData[2].uncertainty + policy.uncertainty + 0.07 - intentShift * 0.03, 0.03, 0.99)
        }
      }

      return {
        ...item,
        phaseDisplay: isZh ? item.phaseLabelZh : item.phase,
        relevance,
        culturalFairness: fairness,
        uncertainty
      }
    })
  }, [explorationIntent, interventionEnabled, isZh, policyMode, repairBudget])

  const highlight = useMemo(() => babelData[Math.min(active, babelData.length - 1)], [active, babelData])

  const policyOptions: Array<{ key: PolicyMode; labelZh: string; labelEn: string }> = [
    { key: 'baseline', labelZh: '基线优先', labelEn: 'Baseline First' },
    { key: 'balanced', labelZh: '均衡策略', labelEn: 'Balanced' },
    { key: 'fairness', labelZh: '公平优先', labelEn: 'Fairness First' }
  ]

  const diagnostic = useMemo(() => {
    const fairnessGap = clamp(highlight.relevance - highlight.culturalFairness, 0, 1)
    const uncertaintyRisk = highlight.uncertainty
    const interventionGain = clamp(
      highlight.culturalFairness - (babelData[2]?.culturalFairness ?? 0.2) + ((babelData[2]?.uncertainty ?? 0.6) - highlight.uncertainty) * 0.4,
      0,
      1
    )

    return [
      {
        label: isZh ? '公平缺口（Fairness Gap）' : 'Fairness Gap',
        value: fairnessGap,
        color: 'bg-zc',
        hint: isZh ? '越低越好：相关性与公平性越接近。' : 'Lower is better: relevance and fairness stay closer.'
      },
      {
        label: isZh ? '不确定风险（Uncertainty Risk）' : 'Uncertainty Risk',
        value: uncertaintyRisk,
        color: 'bg-za',
        hint: isZh ? '越低越好：跨文化解释更稳定。' : 'Lower is better: cross-cultural interpretation is more stable.'
      },
      {
        label: isZh ? '干预增益（Intervention Gain）' : 'Intervention Gain',
        value: interventionGain,
        color: 'bg-zs',
        hint: isZh ? '越高越好：说明修复链路正在生效。' : 'Higher is better: intervention pathway is producing gains.'
      }
    ]
  }, [babelData, highlight.culturalFairness, highlight.relevance, highlight.uncertainty, isZh])

  const tacticalActions = useMemo(
    () => [
      {
        id: 'act1',
        title: isZh ? '提高 PAL 样本预算（Increase PAL budget）' : 'Increase PAL budget',
        done: repairBudget >= 0.6,
        desc: isZh ? '增加高不确定样本的人类标注，优先补齐长尾文化本体。' : 'Allocate more annotation effort to high-uncertainty and long-tail cultures.'
      },
      {
        id: 'act2',
        title: isZh ? '启用干预链路（Enable DDRL intervention）' : 'Enable DDRL intervention',
        done: interventionEnabled,
        desc: isZh ? '开启解纠缠 + OT + 约束回灌，避免“看似多样”但无几何修复。' : 'Enable disentanglement + OT + constraint feedback to avoid cosmetic diversity.'
      },
      {
        id: 'act3',
        title: isZh ? '保持探索意图 > 55%（Exploration > 55%）' : 'Keep exploration > 55%',
        done: explorationIntent > 0.55,
        desc: isZh ? '探索意图偏低会让推荐回到主流文化近邻。' : 'Low exploration pulls recommendations back to mainstream neighbors.'
      }
    ],
    [explorationIntent, interventionEnabled, isZh, repairBudget]
  )

  return (
    <SectionShell
      id="problem"
      title={title}
      subtitle={
        isZh
          ? '借鉴数据叙事网页（Data Essay）方法，本章按“问题-证据-修复”链路展开，不用静态结果表。'
          : 'Like data essays on Pudding, this chapter invites readers to inspect failure modes as a narrative, not a static benchmark table.'
      }
      className="bg-[linear-gradient(180deg,rgba(255,255,255,0.28),transparent_36%)]"
    >
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="reveal-item lg:sticky lg:top-24 lg:h-fit">
          <div className="paper-card rounded-3xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="chapter-chip">{isZh ? '数字巴别塔监测（Digital Babel Monitor）' : 'digital babel monitor'}</p>
              <button
                onClick={() => setStoryAutoPlay((prev) => !prev)}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs font-semibold transition',
                  storyAutoPlay ? 'border-za/35 bg-za/10 text-za' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
                )}
              >
                {isZh ? (storyAutoPlay ? '停止叙事（Stop Story）' : '自动叙事（Auto Story）') : storyAutoPlay ? 'Stop Story' : 'Auto Story'}
              </button>
            </div>
            <h3 className="mt-3 font-display text-3xl text-textMain">{highlight.phaseDisplay}</h3>

            <div className="mt-3 grid gap-2">
              <div className="flex flex-wrap gap-2">
                {policyOptions.map((option) => (
                  <button
                    key={option.key}
                    onClick={() => setPolicyMode(option.key)}
                    className={cn(
                      'rounded-full border px-3 py-1 text-xs font-semibold transition',
                      policyMode === option.key ? 'border-zs/35 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
                    )}
                  >
                    {isZh ? `${option.labelZh}（${option.labelEn}）` : option.labelEn}
                  </button>
                ))}
              </div>

              <label className="block">
                <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                  <span>{isZh ? '探索意图（Exploration Intent）' : 'Exploration Intent'}</span>
                  <span>{(explorationIntent * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={explorationIntent}
                  onChange={(event) => setExplorationIntent(Number(event.target.value))}
                  className="w-full accent-za"
                />
              </label>

              <div className="grid gap-2 sm:grid-cols-[1fr_auto] sm:items-end">
                <label className="block">
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{isZh ? '修复预算（Repair Budget）' : 'Repair Budget'}</span>
                    <span>{(repairBudget * 100).toFixed(0)}%</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.01} value={repairBudget} onChange={(event) => setRepairBudget(Number(event.target.value))} className="w-full accent-zs" />
                </label>

                <button
                  onClick={() => setInterventionEnabled((prev) => !prev)}
                  className={cn(
                    'rounded-full border px-3 py-2 text-xs font-semibold transition',
                    interventionEnabled ? 'border-zs/40 bg-zs/10 text-zs' : 'border-ink/20 bg-white text-textSub hover:text-textMain'
                  )}
                >
                  {isZh ? (interventionEnabled ? '已启用干预（Intervention ON）' : '未启用干预（Intervention OFF）') : interventionEnabled ? 'Intervention ON' : 'Intervention OFF'}
                </button>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2">
              {[
                { label: isZh ? '相关性（Relevance）' : 'Relevance', value: highlight.relevance, color: 'bg-zc' },
                { label: isZh ? '公平性（Fairness）' : 'Fairness', value: highlight.culturalFairness, color: 'bg-zs' },
                { label: isZh ? '不确定性（Uncertainty）' : 'Uncertainty', value: highlight.uncertainty, color: 'bg-za' }
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

            <div className="mt-4 h-[250px] rounded-2xl border border-ink/15 bg-white p-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={babelData} margin={{ left: 2, right: 2, top: 8, bottom: 4 }}>
                  <defs>
                    <linearGradient id="rel" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ea4335" stopOpacity={0.78} />
                      <stop offset="100%" stopColor="#ea4335" stopOpacity={0.08} />
                    </linearGradient>
                    <linearGradient id="fair" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#188038" stopOpacity={0.76} />
                      <stop offset="100%" stopColor="#188038" stopOpacity={0.08} />
                    </linearGradient>
                    <linearGradient id="unc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#1a73e8" stopOpacity={0.68} />
                      <stop offset="100%" stopColor="#1a73e8" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(70,80,100,.22)" strokeDasharray="3 3" />
                  <XAxis dataKey="phaseDisplay" tick={{ fill: '#576178', fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#576178', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#fffdf7', border: '1px solid rgba(60,70,90,.25)' }} />
                  <Area type="monotone" dataKey="relevance" stroke="#ea4335" fill="url(#rel)" strokeWidth={2} />
                  <Area type="monotone" dataKey="culturalFairness" stroke="#188038" fill="url(#fair)" strokeWidth={2} />
                  <Area type="monotone" dataKey="uncertainty" stroke="#1a73e8" fill="url(#unc)" strokeWidth={2} />
                  <ReferenceDot x={highlight.phaseDisplay} y={highlight.relevance} fill="#232938" r={4} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 grid gap-2">
              {diagnostic.map((item) => (
                <div key={item.label} className="rounded-xl border border-ink/15 bg-white px-3 py-2">
                  <div className="mb-1 flex items-center justify-between text-xs text-textSub">
                    <span>{item.label}</span>
                    <span>{(item.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-ink/10">
                    <div className={cn('h-full rounded-full', item.color)} style={{ width: `${item.value * 100}%` }} />
                  </div>
                  <p className="mt-1 text-[11px] text-textSub">{item.hint}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="relative space-y-4">
          <div className="absolute bottom-0 left-[15px] top-0 hidden w-px bg-ink/12 md:block" />

          <div className="reveal-item rounded-3xl border border-zs/25 bg-zs/5 p-4 md:ml-8">
            <p className="chapter-chip">{isZh ? '修复动作清单（Repair Actions）' : 'repair actions'}</p>
            <div className="mt-2 space-y-2 text-sm text-textSub">
              {tacticalActions.map((action) => (
                <p key={action.id} className={action.done ? 'text-zs' : ''}>
                  {action.done ? '●' : '○'} {action.title} · {action.desc}
                </p>
              ))}
            </div>
          </div>

          {chapters.map((chapter, index) => (
            <article
              key={chapter.title}
              onMouseEnter={() => setActive(index)}
              onClick={() => setActive(index)}
              className={cn(
                'reveal-item relative cursor-pointer rounded-3xl border p-6 transition md:ml-8',
                active === index ? 'border-za/40 bg-za/5 shadow-[0_10px_26px_rgba(26,115,232,0.14)]' : 'paper-card hover:border-ink/35'
              )}
            >
              <span className="absolute -left-[28px] top-8 hidden h-3.5 w-3.5 rounded-full border border-ink/20 bg-white md:block" />
              <span className="chapter-chip">{isZh ? `章节 ${String(index + 1).padStart(2, '0')}（Chapter ${String(index + 1).padStart(2, '0')}）` : `chapter ${String(index + 1).padStart(2, '0')}`}</span>
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
