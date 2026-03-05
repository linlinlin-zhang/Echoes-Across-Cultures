'use client'

import { useState } from 'react'

import { SectionShell } from '@/components/layout/section-shell'

export function EthicsSection({ title }: { title: string }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [submitted, setSubmitted] = useState(false)

  return (
    <SectionShell
      id="ethics"
      title={title}
      subtitle="The interface makes uncertainty and ontology boundaries visible, so cultural recommendation remains accountable and negotiable."
    >
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-4 rounded-3xl paper-card p-6">
          <p className="font-display text-3xl leading-tight text-textMain md:text-4xl">
            No universal listener model exists.
            <br />
            Cross-cultural intelligence must stay revisable.
          </p>

          {[
            'Report ontology gaps explicitly instead of hiding them behind aggregate benchmark scores.',
            'Track minority exposure and publish correction loops with transparent criteria.',
            'Expose uncertainty regions in UI so users can see where model confidence is weak.'
          ].map((line) => (
            <div key={line} className="note-card text-sm leading-relaxed text-textSub">
              {line}
            </div>
          ))}

          <div className="rounded-2xl border border-zs/30 bg-zs/10 p-4 text-sm text-textMain">
            ISMIR-ready reporting checklist: dataset provenance, annotation governance, ontology expansion protocol, fairness trade-off analysis.
          </div>
        </div>

        <form
          className="rounded-3xl paper-card p-6"
          onSubmit={(event) => {
            event.preventDefault()
            setSubmitted(true)
          }}
        >
          <h3 className="font-display text-2xl text-textMain">Collaboration Console</h3>
          <p className="mb-4 text-sm text-textSub">Co-design culture concepts, labeling rubrics, and evaluation audits with us.</p>

          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Name</span>
              <input value={name} onChange={(event) => setName(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Email</span>
              <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Message</span>
              <textarea rows={4} value={message} onChange={(event) => setMessage(event.target.value)} className="w-full rounded-xl border border-ink/20 bg-white px-3 py-2 text-sm text-textMain" required />
            </label>

            <button type="submit" className="rounded-full bg-gradient-to-r from-zc to-za px-4 py-2 text-sm font-semibold text-white">
              Send Collaboration Request
            </button>
          </div>

          {submitted ? (
            <p className="mt-3 rounded-lg border border-zs/30 bg-zs/10 px-3 py-2 text-xs text-zs">
              Thanks, {name || 'researcher'}! Demo mode: this form currently stores no external data.
            </p>
          ) : null}
        </form>
      </div>
    </SectionShell>
  )
}