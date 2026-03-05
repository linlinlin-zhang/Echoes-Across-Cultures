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
      subtitle="Research interfaces are also political interfaces. We expose assumptions, uncertainty, and cultural blind spots by design."
    >
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-4 rounded-3xl panel-deep p-6">
          <p className="font-display text-3xl leading-tight text-textMain md:text-4xl">
            No universal listener model exists.
            <br />
            Our system must remain negotiable.
          </p>

          {[
            'We document ontology gaps explicitly instead of hiding them behind aggregate metrics.',
            'We report minority exposure and invite community-defined correction loops.',
            'We preserve uncertainty in the UI so users can see where the model is weak or culturally underfit.'
          ].map((line) => (
            <div key={line} className="rounded-2xl border border-white/10 bg-black/25 p-4 text-sm leading-relaxed text-textSub">
              {line}
            </div>
          ))}

          <div className="rounded-2xl border border-zs/35 bg-zs/10 p-4 text-sm text-textMain">
            Publication note: include dataset provenance, annotation governance, ontology expansion protocol, and fairness trade-off analysis.
          </div>
        </div>

        <form
          className="rounded-3xl panel-deep p-6"
          onSubmit={(event) => {
            event.preventDefault()
            setSubmitted(true)
          }}
        >
          <h3 className="font-display text-2xl text-textMain">Collaboration Console</h3>
          <p className="mb-4 text-sm text-textSub">Invite us to co-design cultural concept nodes, annotation rubrics, and evaluation audits.</p>

          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Name</span>
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Email</span>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <label className="block">
              <span className="mb-1 block font-mono text-[11px] uppercase tracking-[0.14em] text-textSub">Message</span>
              <textarea
                rows={4}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <button
              type="submit"
              className="rounded-full bg-gradient-to-r from-zc to-za px-4 py-2 text-sm font-semibold text-white shadow-[0_0_18px_rgba(165,94,234,0.4)]"
            >
              Send Collaboration Request
            </button>
          </div>

          {submitted ? (
            <p className="mt-3 rounded-lg border border-zs/30 bg-zs/10 px-3 py-2 text-xs text-zs">
              Thanks, {name || 'researcher'}! Demo mode enabled: this form currently stores no external data.
            </p>
          ) : null}
        </form>
      </div>
    </SectionShell>
  )
}