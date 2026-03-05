'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

import { copyByLocale, sectionIds, type SectionId } from '@/data/copy'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import { TopNav } from '@/components/layout/top-nav'
import { SideNav } from '@/components/layout/side-nav'
import { HeroSection } from '@/components/sections/hero-section'
import { ProblemSection } from '@/components/sections/problem-section'
import { ArchitectureSection } from '@/components/sections/architecture-section'
import { GalaxySection } from '@/components/sections/galaxy-section'
import { LabSection } from '@/components/sections/lab-section'
import { PalSection } from '@/components/sections/pal-section'
import { ResultsSection } from '@/components/sections/results-section'
import { EthicsSection } from '@/components/sections/ethics-section'

gsap.registerPlugin(ScrollTrigger)

export function SoundscapePage() {
  const [activeSection, setActiveSection] = useState<SectionId>('hero')
  const { locale, highContrast, reduceMotion } = useAccessibility()
  const copy = useMemo(() => copyByLocale[locale], [locale])

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (!visible.length) return
        const id = visible[0].target.getAttribute('data-section-id') as SectionId | null
        if (id && sectionIds.includes(id)) setActiveSection(id)
      },
      { threshold: [0.3, 0.5, 0.7] }
    )

    document.querySelectorAll<HTMLElement>('[data-section-id]').forEach((section) => observer.observe(section))
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (reduceMotion) {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill())
      return
    }

    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>('.reveal-item').forEach((item) => {
        gsap.fromTo(
          item,
          { opacity: 0, y: 20 },
          {
            opacity: 1,
            y: 0,
            duration: 0.6,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: item,
              start: 'top 84%',
              end: 'bottom 16%',
              toggleActions: 'play none none reverse'
            }
          }
        )
      })
    })
    return () => ctx.revert()
  }, [reduceMotion])

  const scrollToSection = (id: SectionId) => {
    const element = document.getElementById(id)
    if (!element) return
    element.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' })
  }

  return (
    <div className="relative min-h-screen text-textMain">
      <div className="star-fog" />
      <div className="noise-mask" />
      <div className="pointer-events-none fixed inset-0 z-[2] opacity-20 grid-hud" />

      <TopNav labels={copy.nav} activeSection={activeSection} onNavigate={scrollToSection} />
      <SideNav labels={copy.nav} activeSection={activeSection} onNavigate={scrollToSection} />

      <main className="relative z-10">
        <HeroSection
          title={copy.heroTitle}
          lead={copy.heroLead}
          hint={copy.heroHint}
          ctaPrimary={copy.ctaPrimary}
          ctaSecondary={copy.ctaSecondary}
          onNavigate={(id) => scrollToSection(id)}
        />
        <ProblemSection title={copy.sections.problemTitle} />
        <ArchitectureSection title={copy.sections.architectureTitle} />
        <GalaxySection title={copy.sections.galaxyTitle} />
        <LabSection title={copy.sections.labTitle} />
        <PalSection title={copy.sections.palTitle} />
        <ResultsSection title={copy.sections.resultsTitle} />
        <EthicsSection title={copy.sections.ethicsTitle} />
      </main>

      <motion.footer className="relative z-10 border-t border-ink/15 bg-white/75 px-4 py-8 md:px-10" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 text-xs text-textSub">
          <p>Soundscape Without Borders · chapterized editorial interaction</p>
          <p>Accessibility: {highContrast ? 'High Contrast On' : 'Standard Contrast'} · {reduceMotion ? 'Reduced Motion On' : 'Motion On'}</p>
        </div>
      </motion.footer>
    </div>
  )
}