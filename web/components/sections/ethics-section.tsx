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
      subtitle="Cultural recommendation systems must be accountable to communities, transparent about limitations, and open to plural epistemologies."
    >
      <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-4 rounded-3xl border border-white/10 bg-black/30 p-6">
          {[
            'We reject one-size-fits-all cultural embeddings and document ontology gaps explicitly.',
            'We expose minority representation metrics and integrate participatory correction loops.',
            'We preserve uncertainty signals instead of masking low-confidence cross-cultural mappings.'
          ].map((line) => (
            <p key={line} className="font-body text-base leading-relaxed text-textSub">
              {line}
            </p>
          ))}

          <div className="rounded-2xl border border-zs/30 bg-zs/10 p-4 text-sm text-textMain">
            Ethics checkpoint: Always report dataset coverage, weak-supervision boundaries, and cultural ontology blind spots in publications.
          </div>
        </div>

        <form
          className="rounded-3xl border border-white/10 bg-black/30 p-6"
          onSubmit={(event) => {
            event.preventDefault()
            setSubmitted(true)
          }}
        >
          <h3 className="font-display text-2xl text-textMain">Collaboration Contact</h3>
          <p className="mb-4 text-sm text-textSub">Co-design new cultural concepts, annotation protocols, or evaluation criteria.</p>

          <div className="space-y-3">
            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Name</span>
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Email</span>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-textSub">Message</span>
              <textarea
                rows={4}
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                className="w-full rounded-xl border border-white/20 bg-black/35 px-3 py-2 text-sm text-textMain"
                required
              />
            </label>

            <button type="submit" className="rounded-full bg-za px-4 py-2 text-sm font-semibold text-white">
              Send Message
            </button>
          </div>

          {submitted ? (
            <p className="mt-3 rounded-lg border border-zs/30 bg-zs/10 px-3 py-2 text-xs text-zs">
              Thanks, {name || 'researcher'}! This demo uses placeholder submission flow.
            </p>
          ) : null}
        </form>
      </div>
    </SectionShell>
  )
}
