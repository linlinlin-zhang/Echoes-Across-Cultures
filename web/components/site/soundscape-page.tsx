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
  const [pointer, setPointer] = useState({ x: -240, y: -240 })
  const { locale, highContrast, reduceMotion } = useAccessibility()
  const copy = useMemo(() => copyByLocale[locale], [locale])
  const isZh = locale === 'zh'

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
          { opacity: 0, y: 26 },
          {
            opacity: 1,
            y: 0,
            duration: 0.68,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: item,
              start: 'top 88%',
              end: 'bottom 18%',
              toggleActions: 'play none none reverse'
            }
          }
        )
      })
    })
    return () => ctx.revert()
  }, [reduceMotion])

  useEffect(() => {
    if (reduceMotion) return
    const onMove = (event: MouseEvent) => {
      setPointer({ x: event.clientX, y: event.clientY })
    }
    window.addEventListener('mousemove', onMove, { passive: true })
    return () => window.removeEventListener('mousemove', onMove)
  }, [reduceMotion])

  const scrollToSection = (id: SectionId) => {
    const element = document.getElementById(id)
    if (!element) return
    element.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' })
  }

  return (
    <div className="relative min-h-screen bg-deepGradient text-textMain">
      <div className="star-fog" />
      <div className="noise-mask" />
      <div className="pointer-events-none fixed inset-0 z-[2] opacity-24 grid-hud" />
      <motion.div
        className="cursor-blob"
        animate={{ x: pointer.x - 130, y: pointer.y - 130 }}
        transition={{ type: 'spring', stiffness: 110, damping: 24, mass: 0.8 }}
      />

      <TopNav brand={copy.brand} labels={copy.nav} activeSection={activeSection} onNavigate={scrollToSection} />
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
          <p>{isZh ? '声界无疆 · 艺术馆式数据叙事交互界面' : 'Soundscape Without Borders · gallery-style data narrative interface'}</p>
          <p>
            {isZh
              ? `无障碍：${highContrast ? '高对比（High Contrast）已开启' : '标准对比'} · ${reduceMotion ? '减少动画（Reduced Motion）已开启' : '动态模式'}`
              : `Accessibility: ${highContrast ? 'High Contrast On' : 'Standard Contrast'} · ${reduceMotion ? 'Reduced Motion On' : 'Motion On'}`}
          </p>
        </div>
      </motion.footer>
    </div>
  )
}
