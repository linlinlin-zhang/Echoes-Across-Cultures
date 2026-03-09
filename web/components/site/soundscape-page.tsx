'use client'

import dynamic from 'next/dynamic'
import { useEffect, useMemo, useState, type ComponentType } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, BrainCircuit, GaugeCircle, Network, ShieldCheck, SlidersHorizontal, Users, Workflow } from 'lucide-react'

import { TopNav, type NavItem } from '@/components/layout/top-nav'
import { useAccessibility } from '@/components/providers/accessibility-provider'
import type { Locale } from '@/components/ui/types'
import { copyByLocale } from '@/data/copy'
import { cultureNodes, otDemoRoutes, palUncertaintyGrid, songPoints } from '@/data/mock-data'
import { cn } from '@/lib/utils'

const RecommendationDemo = dynamic(() => import('@/components/visuals/recommendation-demo').then((mod) => mod.RecommendationDemo), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const DisentanglementLab = dynamic(() => import('@/components/visuals/disentanglement-lab').then((mod) => mod.DisentanglementLab), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const CultureGalaxyGraph = dynamic(() => import('@/components/visuals/culture-galaxy-graph').then((mod) => mod.CultureGalaxyGraph), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const PalInterface = dynamic(() => import('@/components/visuals/pal-interface').then((mod) => mod.PalInterface), {
  ssr: false,
  loading: () => <WorkbenchLoading />
})

const pageSectionIds = ['experience', 'participate', 'notes'] as const

type PageSectionId = (typeof pageSectionIds)[number]
type WorkbenchTabId = 'lab' | 'alignment' | 'pal'

type WorkbenchTabMeta = {
  id: WorkbenchTabId
  label: string
  shortLabel: string
  eyebrow: string
  title: string
  summary: string
  outcomes: string[]
  prompt: string
}

type PageCopy = {
  nav: Record<PageSectionId, string>
  experienceEyebrow: string
  experienceTitle: string
  experienceLead: string
  experienceButton: string
  experienceStats: Array<{ label: string; value: string }>
  nextTitle: string
  nextLead: string
  participateEyebrow: string
  participateTitle: string
  participateLead: string
  notesEyebrow: string
  notesTitle: string
  notesLead: string
  whyTitle: string
  whyItems: string[]
  systemTitle: string
  systemSteps: string[]
  guardrailsTitle: string
  guardrailsItems: string[]
  footer: string
  statusPrefix: string
  workbenchTabs: WorkbenchTabMeta[]
}

const pageCopyByLocale: Record<Locale, PageCopy> = {
  zh: {
    nav: {
      experience: '立即体验',
      participate: '继续参与',
      notes: '理解原理'
    },
    experienceEyebrow: 'Experience the recommender first',
    experienceTitle: '先试推荐，再理解系统。',
    experienceLead:
      '先选一个听歌场景，立即比较基线与 DDRL 结果。觉得有意思，再继续调潜变量、看文化对齐、进入 PAL。',
    experienceButton: '继续参与系统',
    experienceStats: [
      { label: '可试验样本', value: `${songPoints.length}` },
      { label: '文化节点', value: `${cultureNodes.length}` },
      { label: 'OT 路径', value: `${otDemoRoutes.length}` },
      { label: '高不确定样本', value: `${palUncertaintyGrid.filter((item) => item.value >= 0.6).length}` }
    ],
    nextTitle: '如果推荐结果让你有感觉，下一步就不是“继续看介绍”，而是继续参与。',
    nextLead: '下面三个面板分别对应调参、对齐和人工协作，让用户从“看结果”进入“影响系统”。',
    participateEyebrow: 'Participate beyond the recommendation',
    participateTitle: '继续参与系统，而不只是旁观它。',
    participateLead: '推荐只是第一步。真正的参与感来自继续调参、查看文化对齐关系，并用 PAL 把人的判断重新带回系统。',
    notesEyebrow: 'Research brief',
    notesTitle: '最后再看方法、问题与边界。',
    notesLead: '把研究说明收在第三段，只保留足够支撑理解的内容，不再让它盖过真实体验。',
    whyTitle: '为什么需要它',
    whyItems: [
      '语义坍缩：不同文化语法被压成单一距离，推荐结果表面统一，实际失真。',
      '情感漂移：跨文化迁移共享标签，却不共享语境，容易把情绪含义推荐错。',
      '曝光不平等：少数文化长期被挤出 Top-N，探索意图也会被主流邻域吞掉。'
    ],
    systemTitle: '系统如何工作',
    systemSteps: [
      'A. 用 CultureMERT 主干抽取更稳定的音频表征。',
      'B. 把内容 zc、文化 zs、情感 za 显式解耦。',
      'C. 用泄漏抑制约束减少因子串扰。',
      'D. 用最优传输把跨文化偏好迁移变成可见路线。',
      'E. 用 PAL 把高不确定样本重新送回人工协作。'
    ],
    guardrailsTitle: '哪些边界必须可见',
    guardrailsItems: [
      '不确定区必须公开显示，而不是被总分隐藏。',
      '少数文化曝光与修正回路必须持续被追踪。',
      '人工标注、本体扩展和协作入口必须属于系统主流程。'
    ],
    footer: '声界无疆 · 先体验推荐，再逐步理解系统',
    statusPrefix: '无障碍状态',
    workbenchTabs: [
      {
        id: 'lab',
        label: '潜空间实验室',
        shortLabel: '实验室',
        eyebrow: 'Latent lab',
        title: '继续调 zc / zs / za，亲手影响推荐背后的潜变量。',
        summary: '把推荐体验继续推进到参数层。这里不再只是看结果，而是直接调三因子，观察音色、节奏、情感和文化相似度如何联动。',
        outcomes: ['实时频谱与状态反馈', '音色迁移试听', '参数快照与随机组合'],
        prompt: '适合在看完推荐结果后，追问“这些变化到底是怎么被调出来的”。'
      },
      {
        id: 'alignment',
        label: '文化对齐图',
        shortLabel: '对齐',
        eyebrow: 'Culture alignment',
        title: '查看推荐为什么能跨文化跳转，而不是把它当成黑箱。',
        summary: '在情感对齐与结构对齐之间切换，检索文化节点、看邻域关系，并追踪最优传输路径如何连接不同传统。',
        outcomes: ['节点搜索与选择', '对齐模式切换', 'OT 路线高亮'],
        prompt: '适合把“推荐结果”继续追到“文化关系结构”这一层。'
      },
      {
        id: 'pal',
        label: 'PAL 标注台',
        shortLabel: 'PAL',
        eyebrow: 'Participatory loop',
        title: '把人的判断重新送回系统，让参与感真正落地。',
        summary: '直接处理高不确定样本、提交理由、扩展文化概念，并看 PAL 回合怎样改变覆盖度与风险。',
        outcomes: ['不确定热区筛选', '专家标注与理由输入', '概念扩展与回合指标'],
        prompt: '适合把“我喜欢什么”升级成“我也在影响系统如何学习”。'
      }
    ]
  },
  en: {
    nav: {
      experience: 'Experience',
      participate: 'Participate',
      notes: 'Research'
    },
    experienceEyebrow: 'Experience the recommender first',
    experienceTitle: 'Try the recommendation first, understand the system second.',
    experienceLead:
      'Start with a listening scenario and compare baseline against DDRL immediately. If that creates curiosity, continue into latent controls, cultural alignment, and PAL.',
    experienceButton: 'Continue participating',
    experienceStats: [
      { label: 'Usable samples', value: `${songPoints.length}` },
      { label: 'Culture nodes', value: `${cultureNodes.length}` },
      { label: 'OT routes', value: `${otDemoRoutes.length}` },
      { label: 'High-uncertainty samples', value: `${palUncertaintyGrid.filter((item) => item.value >= 0.6).length}` }
    ],
    nextTitle: 'If the recommendation result creates curiosity, the next step should be participation, not more introduction.',
    nextLead: 'These three panels move the user from watching outcomes to actively influencing the system.',
    participateEyebrow: 'Participate beyond the recommendation',
    participateTitle: 'Keep participating in the system instead of only observing it.',
    participateLead: 'Recommendation is only the first touchpoint. Real participation comes from tuning latent factors, inspecting cultural alignment, and feeding human judgement back through PAL.',
    notesEyebrow: 'Research brief',
    notesTitle: 'Only after that should the method and guardrails step in.',
    notesLead: 'The explanatory layer is compressed into the third section so it supports understanding without overpowering the experience.',
    whyTitle: 'Why it is needed',
    whyItems: [
      'Semantic collapse: different cultural grammars are flattened into one distance metric.',
      'Affective drift: labels travel across cultures without the context that gives them meaning.',
      'Exposure inequality: minority traditions keep getting pushed out of the top-N.'
    ],
    systemTitle: 'How the system works',
    systemSteps: [
      'A. A CultureMERT backbone extracts more stable audio representations.',
      'B. Content zc, culture zs, and affect za are explicitly disentangled.',
      'C. Leakage suppression keeps factors from contaminating each other.',
      'D. Optimal transport makes cross-cultural preference shifts visible.',
      'E. PAL routes high-uncertainty samples back into human collaboration.'
    ],
    guardrailsTitle: 'Which boundaries stay visible',
    guardrailsItems: [
      'Uncertainty zones must stay exposed instead of hidden behind one score.',
      'Minority exposure and correction loops must stay measurable.',
      'Annotation, ontology expansion, and collaboration must remain first-class parts of the system.'
    ],
    footer: 'Soundscape Without Borders · experience recommendation first, understand the system second',
    statusPrefix: 'Accessibility',
    workbenchTabs: [
      {
        id: 'lab',
        label: 'Latent Lab',
        shortLabel: 'Lab',
        eyebrow: 'Latent lab',
        title: 'Keep tuning zc / zs / za and directly influence the factors behind the recommendations.',
        summary: 'Push the recommendation experience deeper into the parameter layer. Instead of only inspecting results, users can tune the three factors and observe how timbre, rhythm, affect, and culture similarity move together.',
        outcomes: ['Realtime spectrum feedback', 'Playable transfer examples', 'Snapshots and surprise presets'],
        prompt: 'Best for asking what actually changed underneath the recommendation result.'
      },
      {
        id: 'alignment',
        label: 'Culture Alignment',
        shortLabel: 'Align',
        eyebrow: 'Culture alignment',
        title: 'See why the recommendation can move across cultures instead of treating it like a black box.',
        summary: 'Switch between emotional and structural alignment, inspect neighborhoods, search culture nodes, and trace OT routes between traditions.',
        outcomes: ['Searchable nodes', 'Alignment mode switching', 'Highlighted OT routes'],
        prompt: 'Best for following recommendation results back to the cultural relationship structure.'
      },
      {
        id: 'pal',
        label: 'PAL Console',
        shortLabel: 'PAL',
        eyebrow: 'Participatory loop',
        title: 'Put human judgement back into the learning loop and make participation real.',
        summary: 'Work through high-uncertainty samples, submit rationales, expand concepts, and inspect how PAL rounds change coverage and risk.',
        outcomes: ['Hotspot triage', 'Annotation and rationale input', 'Concept expansion and round metrics'],
        prompt: 'Best for turning “I like this” into “I can influence how the system learns.”'
      }
    ]
  }
}

const tabAccentClass: Record<WorkbenchTabId, string> = {
  lab: 'border-zc/40 bg-zc/10 text-zc',
  alignment: 'border-zs/40 bg-zs/10 text-zs',
  pal: 'border-ink/25 bg-ink/5 text-textMain'
}

const workbenchComponentByTab: Record<WorkbenchTabId, ComponentType> = {
  lab: DisentanglementLab,
  alignment: CultureGalaxyGraph,
  pal: PalInterface
}

function WorkbenchLoading() {
  return <div className="min-h-[540px] animate-pulse rounded-[28px] bg-white/80" />
}

export function SoundscapePage() {
  const [activeSection, setActiveSection] = useState<PageSectionId>('experience')
  const [activeTab, setActiveTab] = useState<WorkbenchTabId>('lab')
  const { locale, highContrast, reduceMotion } = useAccessibility()

  const chromeCopy = useMemo(() => copyByLocale[locale], [locale])
  const copy = useMemo(() => pageCopyByLocale[locale], [locale])

  const navItems: NavItem[] = useMemo(
    () => [
      { id: 'experience', label: copy.nav.experience },
      { id: 'participate', label: copy.nav.participate },
      { id: 'notes', label: copy.nav.notes }
    ],
    [copy.nav.experience, copy.nav.notes, copy.nav.participate]
  )

  const tabIcons: Record<WorkbenchTabId, ComponentType<{ size?: number | string }>> = {
    lab: SlidersHorizontal,
    alignment: Network,
    pal: Users
  }

  const activeTabMeta = useMemo(() => copy.workbenchTabs.find((item) => item.id === activeTab) ?? copy.workbenchTabs[0], [activeTab, copy.workbenchTabs])
  const ActiveWorkbench = workbenchComponentByTab[activeTab]

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (!visible.length) return
        const id = visible[0].target.getAttribute('data-page-section-id') as PageSectionId | null
        if (id && pageSectionIds.includes(id)) setActiveSection(id)
      },
      { threshold: [0.25, 0.55, 0.8] }
    )

    document.querySelectorAll<HTMLElement>('[data-page-section-id]').forEach((section) => observer.observe(section))
    return () => observer.disconnect()
  }, [])

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (!element) return

    const headerOffset = window.innerWidth < 768 ? 138 : 120
    const top = element.getBoundingClientRect().top + window.scrollY - headerOffset

    window.scrollTo({
      top: Math.max(0, top),
      behavior: reduceMotion ? 'auto' : 'smooth'
    })
  }

  const openWorkbench = (id: WorkbenchTabId) => {
    setActiveTab(id)
    scrollToSection('participate')
  }

  const fadeUp = reduceMotion
    ? {}
    : {
        initial: { opacity: 0, y: 20 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, amount: 0.25 },
        transition: { duration: 0.45, ease: 'easeOut' }
      }

  return (
    <div className="relative min-h-screen bg-deepGradient text-textMain">
      <div className="star-fog" />
      <div className="noise-mask" />
      <div className="pointer-events-none fixed inset-0 z-[2] opacity-20 grid-hud" />

      <TopNav brand={chromeCopy.brand} items={navItems} activeItem={activeSection} onNavigate={scrollToSection} />

      <main className="relative z-10 pb-16 pt-28 md:pt-32">
        <section id="experience" data-page-section-id="experience" className="px-4 pb-8 pt-2 md:px-10">
          <div className="mx-auto max-w-7xl">
            <motion.div {...fadeUp} className="mb-3 flex flex-wrap items-start justify-between gap-3">
              <div className="max-w-3xl">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="sticker">{copy.experienceEyebrow}</span>
                  <span className="chapter-chip">{chromeCopy.subtitle}</span>
                </div>
                <h1 className="mt-2 max-w-4xl font-display text-2xl font-semibold leading-[0.96] text-textMain md:text-4xl">{copy.experienceTitle}</h1>
                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-textSub md:text-base">{copy.experienceLead}</p>
              </div>

              <button
                onClick={() => scrollToSection('participate')}
                className="inline-flex items-center gap-2 rounded-full border border-ink/20 bg-white px-4 py-2 text-sm font-semibold text-textMain transition hover:border-zs/35 hover:bg-zs/5"
              >
                {copy.experienceButton}
                <ArrowRight size={15} />
              </button>
            </motion.div>

            <motion.div {...fadeUp} className="rounded-[36px] panel-glass p-3 md:p-4">
              <RecommendationDemo />
            </motion.div>

            <motion.div {...fadeUp} className="mt-4 flex flex-wrap gap-2">
              {copy.experienceStats.map((item) => (
                <div key={item.label} className="rounded-full border border-ink/15 bg-white/85 px-3 py-2 text-sm text-textSub">
                  <span className="font-mono uppercase tracking-[0.12em]">{item.label}</span>
                  <span className="ml-2 font-semibold text-textMain">{item.value}</span>
                </div>
              ))}
            </motion.div>

            <motion.div {...fadeUp} className="mt-5">
              <p className="chapter-chip">{copy.nextTitle}</p>
              <p className="mt-2 max-w-3xl text-sm leading-relaxed text-textSub">{copy.nextLead}</p>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {copy.workbenchTabs.map((tab) => {
                  const Icon = tabIcons[tab.id]
                  return (
                    <button
                      key={tab.id}
                      onClick={() => openWorkbench(tab.id)}
                      className="rounded-3xl border bg-white/88 p-5 text-left transition hover:border-ink/30 hover:bg-white"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-za ring-1 ring-ink/10">
                          <Icon size={18} />
                        </div>
                        <span className="chapter-chip">{tab.eyebrow}</span>
                      </div>
                      <h2 className="mt-4 font-display text-2xl leading-tight text-textMain">{tab.label}</h2>
                      <p className="mt-2 text-sm leading-relaxed text-textSub">{tab.summary}</p>
                      <div className="mt-4 inline-flex items-center gap-2 text-sm font-semibold text-za">
                        {locale === 'zh' ? '打开面板' : 'Open panel'}
                        <ArrowRight size={14} />
                      </div>
                    </button>
                  )
                })}
              </div>
            </motion.div>
          </div>
        </section>

        <section id="participate" data-page-section-id="participate" className="px-4 py-8 md:px-10">
          <div className="mx-auto max-w-7xl">
            <motion.div {...fadeUp} className="mb-5 max-w-3xl">
              <p className="chapter-chip">{copy.participateEyebrow}</p>
              <h2 className="mt-3 font-display text-4xl font-semibold leading-[0.97] text-textMain md:text-5xl">{copy.participateTitle}</h2>
              <p className="mt-3 text-base leading-relaxed text-textSub">{copy.participateLead}</p>
            </motion.div>

            <motion.div {...fadeUp} className="rounded-[36px] panel-glass p-4 md:p-5">
              <div className="flex gap-2 overflow-x-auto pb-1">
                {copy.workbenchTabs.map((tab) => {
                  const Icon = tabIcons[tab.id]
                  const active = tab.id === activeTab

                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={cn(
                        'inline-flex min-w-fit items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold transition',
                        active ? tabAccentClass[tab.id] : 'border-ink/15 bg-white/80 text-textSub hover:border-ink/30 hover:text-textMain'
                      )}
                    >
                      <Icon size={15} />
                      {tab.label}
                    </button>
                  )
                })}
              </div>

              <div className="mt-5 grid gap-4 xl:grid-cols-[300px_1fr]">
                <div className="rounded-[28px] panel-deep p-5 xl:sticky xl:top-24 xl:h-fit">
                  <span className="sticker">{activeTabMeta.eyebrow}</span>
                  <h3 className="mt-4 font-display text-3xl leading-tight text-textMain">{activeTabMeta.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-textSub">{activeTabMeta.summary}</p>

                  <div className="mt-5 space-y-2">
                    {activeTabMeta.outcomes.map((item) => (
                      <div key={item} className="rounded-2xl border border-ink/15 bg-white/85 px-3 py-2 text-sm text-textMain">
                        {item}
                      </div>
                    ))}
                  </div>

                  <div className="mt-5 rounded-2xl border border-zs/30 bg-zs/10 p-4 text-sm leading-relaxed text-textMain">{activeTabMeta.prompt}</div>
                </div>

                <div className="rounded-[28px] border border-ink/10 bg-white/86 p-2 md:p-3">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeTab}
                      initial={reduceMotion ? undefined : { opacity: 0, y: 10 }}
                      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
                      exit={reduceMotion ? undefined : { opacity: 0, y: -10 }}
                      transition={reduceMotion ? undefined : { duration: 0.24, ease: 'easeOut' }}
                    >
                      <ActiveWorkbench />
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section id="notes" data-page-section-id="notes" className="px-4 pt-8 md:px-10">
          <div className="mx-auto max-w-7xl">
            <motion.div {...fadeUp} className="mb-6 max-w-3xl">
              <p className="chapter-chip">{copy.notesEyebrow}</p>
              <h2 className="mt-3 font-display text-4xl font-semibold leading-[0.98] text-textMain md:text-5xl">{copy.notesTitle}</h2>
              <p className="mt-3 text-base leading-relaxed text-textSub">{copy.notesLead}</p>
            </motion.div>

            <div className="grid gap-4 xl:grid-cols-[1fr_1fr_1fr]">
              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-zc/12 text-zc">
                  <GaugeCircle size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.whyTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.whyItems.map((item) => (
                    <div key={item} className="note-card text-sm leading-relaxed text-textSub">
                      {item}
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-za/12 text-za">
                  <Workflow size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.systemTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.systemSteps.map((item, index) => (
                    <div key={item} className="rounded-2xl border border-ink/15 bg-white/90 p-4">
                      <div className="flex items-start gap-3">
                        <span className="inline-flex h-7 w-7 flex-none items-center justify-center rounded-full bg-ink/5 font-mono text-xs text-textMain">{index + 1}</span>
                        <p className="text-sm leading-relaxed text-textSub">{item}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>

              <motion.div {...fadeUp} className="rounded-[30px] paper-card p-6">
                <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-zs/12 text-zs">
                  <ShieldCheck size={18} />
                </div>
                <h3 className="mt-4 font-display text-2xl text-textMain">{copy.guardrailsTitle}</h3>
                <div className="mt-4 space-y-3">
                  {copy.guardrailsItems.map((item) => (
                    <div key={item} className="note-card text-sm leading-relaxed text-textSub">
                      {item}
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => openWorkbench('pal')}
                  className="mt-5 inline-flex items-center gap-2 rounded-full border border-zs/30 bg-zs/10 px-4 py-2 text-sm font-semibold text-zs transition hover:bg-zs/15"
                >
                  <BrainCircuit size={15} />
                  {locale === 'zh' ? '把你的判断带回 PAL' : 'Bring your judgement into PAL'}
                </button>
              </motion.div>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative z-10 mt-10 border-t border-ink/15 bg-white/75 px-4 py-8 md:px-10">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 text-xs text-textSub">
          <p>{copy.footer}</p>
          <p>
            {locale === 'zh'
              ? `${copy.statusPrefix}：${highContrast ? '高对比已开启' : '标准对比'} · ${reduceMotion ? '减少动画已开启' : '动态模式'}`
              : `${copy.statusPrefix}: ${highContrast ? 'High Contrast On' : 'Standard Contrast'} · ${reduceMotion ? 'Reduced Motion On' : 'Motion On'}`}
          </p>
        </div>
      </footer>
    </div>
  )
}
