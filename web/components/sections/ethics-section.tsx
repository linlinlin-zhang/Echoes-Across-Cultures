'use client'

import { useState } from 'react'

import { SectionShell } from '@/components/layout/section-shell'
import { useAccessibility } from '@/components/providers/accessibility-provider'

export function EthicsSection({ title }: { title: string }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const { locale } = useAccessibility()
  const isZh = locale === 'zh'

  const pledges = isZh
    ? [
        '明确报告本体缺口（Ontology Gaps），而不是被汇总 benchmark 分数掩盖。',
        '持续追踪少数文化曝光，并公开修正回路（Correction Loops）与判据。',
        '在界面中显示不确定性区域（Uncertainty Regions），让用户看到模型薄弱带。'
      ]
    : [
        'Report ontology gaps explicitly instead of hiding them behind aggregate benchmark scores.',
        'Track minority exposure and publish correction loops with transparent criteria.',
        'Expose uncertainty regions in UI so users can see where model confidence is weak.'
      ]

  return (
    <SectionShell
      id="ethics"
      title={title}
      subtitle={
        isZh
          ? '界面将不确定性与本体边界显性化（Visible Boundaries），让跨文化推荐保持可问责、可协商。'
          : 'The interface makes uncertainty and ontology boundaries visible, so cultural recommendation remains accountable and negotiable.'
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-4 rounded-3xl paper-card p-6">
          <p className="font-display text-3xl leading-tight text-textMain md:text-4xl">
            {isZh ? '不存在唯一的“普适听众模型”。' : 'No universal listener model exists.'}
            <br />
            {isZh ? '跨文化智能必须保持可修订（Revisable）。' : 'Cross-cultural intelligence must stay revisable.'}
          </p>

          {pledges.map((line) => (
            <div key={line} className="note-card text-sm leading-relaxed text-textSub">
              {line}
            </div>
          ))}

          <div className="rounded-2xl border border-zs/30 bg-zs/10 p-4 text-sm text-textMain">
            {isZh
              ? 'ISMIR 级报告清单：数据来源（Dataset Provenance）、标注治理（Annotation Governance）、本体扩展协议（Ontology Expansion Protocol）、公平性权衡分析（Fairness Trade-off Analysis）。'
              : 'ISMIR-ready reporting checklist: dataset provenance, annotation governance, ontology expansion protocol, fairness trade-off analysis.'}
          </div>
        </div>

        <form
          className="rounded-3xl paper-card p-6"
          onSubmit={(event) => {
            event.preventDefault()
            setSubmitted(true)
          }}
        >
          <h3 className="font-display text-2xl text-textMain">{isZh ? '协作控制台（Collaboration Console）' : 'Collaboration Console'}</h3>
          <p className="mb-4 text-sm text-textSub">
            {isZh
              ? '与我们共建文化概念（Culture Concepts）、标注规范（Labeling Rubrics）与评测审计（Evaluation Audits）。'
              : 'Co-design culture concepts, labeling rubrics, and evaluation audits with us.'}
          </p>

          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">{isZh ? '姓名（Name）' : 'Name'}</span>
              <input value={name} onChange={(event) => setName(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">{isZh ? '邮箱（Email）' : 'Email'}</span>
              <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">{isZh ? '留言（Message）' : 'Message'}</span>
              <textarea rows={4} value={message} onChange={(event) => setMessage(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <button type="submit" className="rounded-full bg-gradient-to-r from-zc to-za px-4 py-2 text-sm font-semibold text-white">
              {isZh ? '发送协作请求（Send Collaboration Request）' : 'Send Collaboration Request'}
            </button>
          </div>

          {submitted ? (
            <p className="mt-3 rounded-lg border border-zs/30 bg-zs/10 px-3 py-2 text-xs text-zs">
              {isZh
                ? `感谢你，${name || '研究者'}！当前为演示模式（Demo Mode），表单不会写入外部数据库。`
                : `Thanks, ${name || 'researcher'}! Demo mode: this form currently stores no external data.`}
            </p>
          ) : null}
        </form>
      </div>
    </SectionShell>
  )
}
